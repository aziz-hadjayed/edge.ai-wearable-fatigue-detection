import gc
import json
import random
import sys
import traceback
import warnings
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
import optuna
import pandas as pd
# pyrefly: ignore [missing-import]
from reservoirpy.nodes import Reservoir, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import *
import onnx
import onnx.helper
import onnx.numpy_helper
import onnxruntime as ort

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from utils.apply_smote import resample_dataframe
from models.semi_supervised import add_pseudo_labels
# pyrefly: ignore [missing-import]
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 1. CONSTANTES
# ============================================================================
MODEL_NAME = "ESN"
LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES = ["baseline", "activity", "pre_fatigue", "fatigue"]

# ============================================================================
# 2. ESN CORE
# ============================================================================
def _free_memory(*objects):
    """
    Libère explicitement les objets temporaires et force le garbage collector.
    Utile pour éviter la saturation de la RAM lors des folds LOSO ou des trials Optuna.
    """
    for obj in objects:
        del obj
    gc.collect()


def _build_reservoir(params, input_dim):
    """
    Initialise un noeud Reservoir (ESN) avec les hyperparamètres spécifiés.
    Le réservoir transforme les entrées de dimension 'input_dim' en un espace de haute dimension.
    """
    n_reservoir = int(params["n_reservoir"])
    input_scaling = float(params["input_scaling"])
    sparsity = float(params["sparsity"])
    # RC connectivity est l'inverse de la sparsité (pourcentage de connexions actives).
    rc_connectivity = max(0.0, min(1.0, 1.0 - sparsity))
    spectral_radius = float(params["spectral_radius"])

    node = Reservoir(
        units=n_reservoir,
        lr=float(params["leak_rate"]),
        sr=spectral_radius,
        input_scaling=input_scaling,
        input_connectivity=1.0,
        rc_connectivity=rc_connectivity,
        input_dim=input_dim,
        dtype=np.float32,
        seed=int(params.get("random_state", 42)),
        name="reservoir",
    )

    return {
        "node": node,
        "params": dict(params),
        "input_dim": int(input_dim),
    }


def _reservoir_states(window, reservoir):
    """
    Calcule les états internes du réservoir pour une fenêtre de données donnée.
    Le 'washout' permet d'ignorer les premiers états transitoires instables.
    """
    params = reservoir["params"]
    node = reservoir["node"]
    washout = int(params.get("washout", 0))

    if node.initialized:
        node.reset()
    # Passer la fenêtre dans le réservoir pour obtenir la trajectoire des états.
    states = node.run(window.astype(np.float32))
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states.reshape(1, -1)

    # Appliquer le washout : on ne garde que les états après la période de chauffe.
    states = states[washout:] if washout < len(states) else states[-1:]
    return states


def _summarize_states(states, params):
    """
    Agrège la séquence d'états du réservoir en un vecteur de features fixe.
    Plusieurs méthodes (summary) sont possibles : dernier état, moyenne, ou concaténation.
    """
    summary = params.get("state_summary", "last_mean_std")
    last = states[-1]

    if summary == "last":
        return last.astype(np.float32)
    if summary == "last_mean":
        # Concatène le dernier état et la moyenne temporelle des états.
        return np.concatenate([last, states.mean(axis=0)]).astype(np.float32)
    if summary == "last_mean_std":
        # Version riche : dernier état + moyenne + écart-type (capture la dynamique).
        return np.concatenate([last, states.mean(axis=0), states.std(axis=0)]).astype(np.float32)

    raise ValueError(f"state_summary inconnu: {summary}")


def _transform_window(window, reservoir):
    """
    Pipeline complet pour transformer une fenêtre de signaux bruts en features ESN.
    """
    states = _reservoir_states(window, reservoir)
    return _summarize_states(states, reservoir["params"])


# ============================================================================
# 3. DONNEES
# ============================================================================
def _sort_df(df):
    """
    Trie le DataFrame par participant, session et timestamp pour garantir la continuité temporelle.
    """
    if COL_TIMESTAMP in df.columns:
        return df.sort_values([COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP])
    return df.sort_values([COL_PARTICIPANT, COL_SESSION])


def _window_label(y_window):
    """
    Détermine le label unique d'une fenêtre par vote majoritaire sur ses échantillons.
    """
    # Filtrer les valeurs NaN
    valid = y_window[~np.isnan(y_window)]
    if valid.size == 0:
        return -1
    values, counts = np.unique(valid, return_counts=True)
    return int(values[np.argmax(counts)])
    """
    Détermine le label unique d'une fenêtre par vote majoritaire sur ses échantillons.
    """
    # Filter out NaN values
    valid = y_window[~np.isnan(y_window)]
    if valid.size == 0:
        return -1
    values, counts = np.unique(valid, return_counts=True)
    return int(values[np.argmax(counts)])


def extract_esn_windows(df, reservoir, window_size, step_size):
    """
    Découpe les données en fenêtres glissantes et les transforme en features via le réservoir.
    Le découpage respecte les frontières des sessions (pas de chevauchement entre sessions).
    Les fenêtres contenant des labels NaN sont ignorées pour éviter les erreurs de conversion.
    """
    x_all, y_all = [], []
    df = _sort_df(df)

    for _, group in df.groupby([COL_PARTICIPANT, COL_SESSION], sort=False):
        x_raw = group[SIGNAL_COLS].values.astype(np.float32)
        y_raw = group[COL_LABEL].values  # keep original dtype to allow NaN filtering
        # Parcourir les fenêtres possibles
        for start in range(0, len(x_raw) - window_size + 1, step_size):
            end = start + window_size
            y_window = y_raw[start:end]
            # Ignorer les fenêtres contenant des NaN
            if np.isnan(y_window).any():
                continue
            # Conversion en int32 après filtrage
            y_window_int = y_window.astype(np.int32)
            # Transformation reservoirpy de la fenêtre brute.
            x_all.append(_transform_window(x_raw[start:end], reservoir))
            y_all.append(_window_label(y_window_int))

    if not x_all:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int32)
    return np.asarray(x_all, dtype=np.float32), np.asarray(y_all, dtype=np.int32)
    """
    Découpe les données en fenêtres glissantes et les transforme en features via le réservoir.
    Le découpage respecte les frontières des sessions (pas de chevauchement entre sessions).
    Les fenêtres contenant des labels NaN sont ignorées pour éviter les erreurs de conversion.
    """
    x_all, y_all = [], []
    df = _sort_df(df)

    for _, group in df.groupby([COL_PARTICIPANT, COL_SESSION], sort=False):
        x_raw = group[SIGNAL_COLS].values.astype(np.float32)
        y_raw = group[COL_LABEL].values
        # Parcourir les fenêtres possibles
        for start in range(0, len(x_raw) - window_size + 1, step_size):
            end = start + window_size
            y_window = y_raw[start:end]
            # Ignorer les fenêtres contenant des NaN ou des valeurs non entières
            if np.isnan(y_window).any():
                continue
            # Convertir en int32 après le filtrage
            y_window_int = y_window.astype(np.int32)
            x_all.append(_transform_window(x_raw[start:end], reservoir))
            y_all.append(_window_label(y_window_int))

    if not x_all:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int32)
    return np.asarray(x_all, dtype=np.float32), np.asarray(y_all, dtype=np.int32)


def _extract_raw_windows_sample(df, window_size, step_size, n_samples=20):
    """Extrait un echantillon de fenetres BRUTES (avant reservoir) pour la validation ONNX."""
    df = _sort_df(df)
    raw_windows = []
    for _, group in df.groupby([COL_PARTICIPANT, COL_SESSION], sort=False):
        x_raw = group[SIGNAL_COLS].values.astype(np.float32)
        y_raw = group[COL_LABEL].values
        for start in range(0, len(x_raw) - window_size + 1, step_size):
            end = start + window_size
            if np.isnan(y_raw[start:end]).any():
                continue
            raw_windows.append(x_raw[start:end])
            if len(raw_windows) >= n_samples:
                return np.asarray(raw_windows, dtype=np.float32)
    return np.asarray(raw_windows, dtype=np.float32) if raw_windows else np.empty((0, window_size, len(SIGNAL_COLS)), dtype=np.float32)


def first_window_states(df, reservoir, window_size):
    """
    Extrait les états de la toute première fenêtre valide d'un DataFrame.
    Utilisé pour la visualisation de la dynamique du réservoir.
    """
    df = _sort_df(df)
    for _, group in df.groupby([COL_PARTICIPANT, COL_SESSION], sort=False):
        x_raw = group[SIGNAL_COLS].values.astype(np.float32)
        if len(x_raw) >= window_size:
            return _reservoir_states(x_raw[:window_size], reservoir)
    return None


# ============================================================================
# 4. MODELE, OPTUNA
# ============================================================================
def build_model(params):
    """
    Crée le modèle de readout (Ridge Regression) qui sera entraîné sur les états du réservoir.
    """
    return Ridge(ridge=float(params["ridge_alpha"]), name="readout")


def _sample_weights(y_train):
    """
    Calcule des poids pour compenser le déséquilibre des classes (baseline vs fatigue vs activity).
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = dict(zip(classes, weights))
    return np.asarray([weight_map[y] for y in y_train], dtype=np.float32)


def _to_onehot(y, num_classes=None):
    """
    Encode les labels en format One-Hot pour l'entraînement du readout.
    """
    num_classes = num_classes or len(LABEL_MAPPING)
    y_onehot = np.zeros((len(y), num_classes), dtype=np.float32)
    y_onehot[np.arange(len(y)), y.astype(int)] = 1.0
    return y_onehot


def fit_readout(model, x_train, y_train):
    """
    Entraîne le readout Ridge sur les features extraites par le réservoir.
    Applique une pondération par classe pour gérer le déséquilibre du dataset.
    """
    # Utilisation de la racine carrée des poids pour la pondération dans Ridge.
    weights = np.sqrt(_sample_weights(y_train)).reshape(-1, 1)
    x_weighted = x_train * weights
    y_weighted = _to_onehot(y_train) * weights
    model.fit(x_weighted, y_weighted)
    return model
    

def predict_readout(model, x):
    """
    Réalise des prédictions de classes à partir des scores de sortie du readout.
    """
    scores = np.asarray(model.run(x), dtype=np.float32)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    # La classe prédite est celle ayant le score le plus élevé (argmax).
    return np.argmax(scores, axis=1).astype(np.int32)


class _ReadoutProbaWrapper:
    """
    Adapte le Ridge readout de reservoirpy pour exposer une methode predict_proba,
    compatible avec add_pseudo_labels() de semi_supervised.py qui attend soit
    predict_proba (sklearn), soit un predict() retournant des probabilites 2D (Keras).
    Le Ridge de reservoirpy n'a ni l'un ni l'autre nativement (model.run() retourne
    des scores bruts de regression, pas des probabilites).
    """
    def __init__(self, readout_model):
        self.readout_model = readout_model

    def predict_proba(self, x):
        scores = np.asarray(self.readout_model.run(x), dtype=np.float32)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        # Softmax pour convertir les scores bruts en pseudo-probabilites.
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _suggest_esn_params(trial):
    """
    Définit l'espace de recherche des hyperparamètres pour Optuna.
    """
    space = ESN_OPTUNA_SPACE
    return {
        "n_reservoir": trial.suggest_categorical("n_reservoir", space["n_reservoir"]),
        "spectral_radius": trial.suggest_float("spectral_radius", **space["spectral_radius"]),
        "sparsity": trial.suggest_float("sparsity", **space["sparsity"]),
        "leak_rate": trial.suggest_float("leak_rate", **space["leak_rate"]),
        "input_scaling": trial.suggest_float("input_scaling", **space["input_scaling"]),
        "ridge_alpha": trial.suggest_float("ridge_alpha", **space["ridge_alpha"]),
        "washout": trial.suggest_categorical("washout", space["washout"]),
        "state_summary": trial.suggest_categorical("state_summary", space["state_summary"]),
        "random_state": 42,
    }


def optuna_objective(trial, df, sessions, window_size, step_size):
    """
    Fonction objectif pour Optuna : évalue un jeu de paramètres sur quelques sessions.
    Retourne la moyenne du F1-score macro sur les sessions de validation.
    """
    params = _suggest_esn_params(trial)
    scores = []

    for val_part, val_sess in sessions:
        model = None
        try:
            # Séparation Leave-One-Session-Out pour l'optimisation.
            df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
            df_val = df[(df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess)].copy()

            # SMOTE seulement sur df_train, avant le scaling et l'extraction de features ESN.
            df_train = resample_dataframe(df_train, SIGNAL_COLS)

            # Normalisation robuste des signaux.
            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_val[SIGNAL_COLS] = scaler.transform(df_val[SIGNAL_COLS])

            # Extraction des features ESN.
            reservoir = _build_reservoir(params, len(SIGNAL_COLS))
            x_train, y_train = extract_esn_windows(df_train, reservoir, window_size, step_size)
            x_val, y_val = extract_esn_windows(df_val, reservoir, window_size, step_size)
            if len(y_train) == 0 or len(y_val) == 0:
                scores.append(0.0)
                continue

            # Entraînement et évaluation rapide.
            model = build_model(params)
            fit_readout(model, x_train, y_train)
            y_pred = predict_readout(model, x_val)
            scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        except Exception as exc:
            print(f"  [WARN] trial split echoue ({type(exc).__name__}: {exc})")
            scores.append(0.0)
        finally:
            _free_memory(model)

    return float(np.mean(scores)) if scores else 0.0


def optimize_hyperparams(df, n_trials=ESN_OPTUNA_TRIALS):
    """
    Lance l'étude Optuna pour trouver les meilleurs hyperparamètres de l'ESN.
    """
    print(f"\nOPTUNA ESN - {n_trials} trials | {ESN_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
    random.seed(42)
    window_size = WINDOW_CONFIGS["default"]["window_size"]
    step_size = WINDOW_CONFIGS["default"]["step_size"]
    
    # Sélectionner aléatoirement quelques sessions pour accélérer Optuna.
    all_sessions = [tuple(x) for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
    val_sessions = random.sample(all_sessions, min(ESN_OPTUNA_SESSIONS, len(all_sessions)))
    print(f"Sessions Optuna: {val_sessions}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, df, val_sessions, window_size, step_size),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
        gc_after_trial=True,
        catch=(Exception,),
    )

    print(f"\nBest F1-Macro: {study.best_value:.4f}")
    print(f"Best params  : {study.best_params}")
    
    # Sauvegarder les résultats de l'optimisation.
    OPTUNA_PATH_ESN.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_ESN, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return {**MODEL_PARAMS["ESN"], **study.best_params}


# ============================================================================
# 5. VISUALISATION
# ============================================================================
def plot_fold(fold_idx, test_part, test_sess, save_dir, f1_mac, bal_acc, y_true, y_pred, states):
    """
    Génère et sauvegarde des graphiques de diagnostic pour chaque fold LOSO.
    Inclut scores, matrice de confusion et l'énergie des états du réservoir.
    """
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        f"ESN | Fold {fold_idx} - P{test_part} S{test_sess}"
        f" | F1-Macro: {f1_mac:.3f} | Bal.Acc: {bal_acc:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    # 1. Histogramme des scores.
    axes[0].bar(["F1-Macro", "Balanced Acc"], [f1_mac, bal_acc], color=["#3B82F6", "#10B981"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Scores du fold")
    axes[0].grid(True, axis="y", alpha=0.3)

    # 2. Matrice de confusion.
    cm = confusion_matrix(y_true, y_pred)
    labels = [TARGET_NAMES[i] for i in np.unique(np.concatenate([y_true, y_pred]))]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        linewidths=0.5,
        linecolor="gray",
    )
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predit")
    axes[1].set_ylabel("Reel")
    axes[1].tick_params(axis="x", rotation=20)

    # 3. Dynamique du réservoir (énergie moyenne des états).
    if states is not None and len(states) > 0:
        energy = np.mean(np.abs(states), axis=1)
        axes[2].plot(np.arange(len(energy)), energy, color="#F97316", linewidth=2)
        axes[2].set_title("Energie moyenne du reservoir")
        axes[2].set_xlabel("Temps apres washout")
        axes[2].set_ylabel("mean(|state|)")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    plot_path = save_dir / f"fold{fold_idx:02d}_P{test_part}_S{test_sess}_curves.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Courbes sauvegardees: {plot_path}")


# ============================================================================
# 6. MODELE GLOBAL
# ============================================================================
def _compute_classification_metrics(y_true, y_pred):
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_fatigue": float(
            precision_score(y_true, y_pred, labels=[3], average="macro", zero_division=0)
        ),
        "recall_fatigue": float(
            recall_score(y_true, y_pred, labels=[3], average="macro", zero_division=0)
        ),
    }


def _save_edge_comparison_metrics(model_name, edge_metrics):
    curr_metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                content = f.read().strip()
            curr_metrics = json.loads(content) if content else {}
        except (json.JSONDecodeError, ValueError):
            print(f"metrics.json invalide ou vide, reinitialisation: {METRICS_PATH}")
            curr_metrics = {}

    curr_metrics.setdefault(model_name, {})
    curr_metrics[model_name]["edge_comparison"] = edge_metrics

    with open(METRICS_PATH, "w") as f:
        json.dump(curr_metrics, f, indent=4)
    print(f"  Comparaison edge sauvegardee: {METRICS_PATH}")


def _extract_reservoir_weights(reservoir):
    node = reservoir["node"]
    W_in = node.Win.toarray() if hasattr(node.Win, "toarray") else np.asarray(node.Win)
    W = node.W.toarray() if hasattr(node.W, "toarray") else np.asarray(node.W)
    leak_rate = float(node.lr)
    return W_in, W, leak_rate

def _export_c_array_from_bytes(file_path, save_dir, model_name):
    with open(file_path, "rb") as f:
        data = f.read()
    c_array = ", ".join(f"0x{b:02x}" for b in data)
    h_path = save_dir / f"{model_name}.h"
    with open(h_path, "w") as f:
        f.write(f"const unsigned char {model_name}_onnx[] = {{{c_array}}};\n")
        f.write(f"const unsigned int {model_name}_onnx_len = {len(data)};\n")
    print(f"  Header C exporté : {h_path}")

def _validate_onnx_esn(onnx_model, reservoir, readout_model, X_sample_windows):
    """
    Compare les predictions du graphe ONNX a celles du chemin reservoirpy reel
    (meme reservoir, meme readout), sur un echantillon de fenetres deja extraites
    (X_sample_windows : fenetres brutes AVANT resume par le reservoir, shape
    (n_samples, window_size, n_features) — PAS les features deja resumees).
    Affiche le pourcentage d'accord et le MAE, avec un WARNING si l'accord est
    insuffisant (ne bloque pas l'execution, juste avertit).
    """
    ort_session = ort.InferenceSession(onnx_model.SerializeToString())
    input_name = ort_session.get_inputs()[0].name

    # Predictions via le chemin reservoirpy reel (recalcule les features resumees
    # window par window, pour rester coherent avec transform_session_windows/extract_esn_windows)
    y_pred_native = []
    for window in X_sample_windows:
        summary = _transform_window(window, reservoir)
        score = np.asarray(readout_model.run(summary.reshape(1, -1)), dtype=np.float32)
        if score.ndim == 1:
            score = score.reshape(1, -1)
        y_pred_native.append(int(np.argmax(score, axis=1)[0]))
    y_pred_native = np.array(y_pred_native, dtype=np.int32)

    # Predictions via le graphe ONNX
    outputs = ort_session.run(None, {input_name: X_sample_windows.astype(np.float32)})
    proba_onnx = outputs[0]
    label_onnx = outputs[1].astype(np.int32)

    agreement = float(np.mean(y_pred_native == label_onnx)) * 100.0

    # MAE sur les probabilites (proxy de l'ecart numerique, en plus de l'accord des classes)
    onehot_native = np.zeros_like(proba_onnx)
    onehot_native[np.arange(len(y_pred_native)), y_pred_native] = 1.0
    mae = float(np.mean(np.abs(proba_onnx - onehot_native)))

    print(f"  Validation ONNX ESN : {agreement:.1f}% d'accord sur {len(X_sample_windows)} fenetres, MAE={mae:.4f}")
    if agreement < 95.0:
        print(f"  WARNING: accord ONNX/reservoirpy insuffisant ({agreement:.1f}% < 95%) — "
              f"verifier la transposition de W_in/W dans _build_onnx_esn_graph, ou la "
              f"formule de recurrence/agregation.")

    return agreement >= 95.0

def _predict_onnx(onnx_path, X_windows):
    ort_session = ort.InferenceSession(str(onnx_path))
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: X_windows.astype(np.float32)})
    return np.asarray(outputs[1], dtype=np.int32)

def _build_onnx_esn_graph(
    W_in,
    W,
    leak_rate,
    Wout,
    bias,
    window_size,
    n_features,
    n_reservoir,
    num_classes,
    washout,
    state_summary,
    model_name,
):
    # reservoirpy stocke Win (n_reservoir, n_features) et W (n_reservoir, n_reservoir).
    # En convention ligne (MatMul ONNX), il faut x @ Win.T et state @ W.T.
    W_in_T = np.asarray(W_in, dtype=np.float32).T
    W_T = np.asarray(W, dtype=np.float32).T
    Wout = np.asarray(Wout, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32).reshape(-1)

    nodes = []
    initializers = [
        onnx.numpy_helper.from_array(W_in_T, "W_in_T"),
        onnx.numpy_helper.from_array(W_T, "W_T"),
        onnx.numpy_helper.from_array(Wout.astype(np.float32), "Wout"),
        onnx.numpy_helper.from_array(bias.astype(np.float32), "bias"),
        onnx.numpy_helper.from_array(np.asarray([1.0 - float(leak_rate)], dtype=np.float32), "one_minus_leak"),
        onnx.numpy_helper.from_array(np.asarray([float(leak_rate)], dtype=np.float32), "leak"),
        onnx.numpy_helper.from_array(np.zeros((1, n_reservoir), dtype=np.float32), "state_init"),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "axis_0"),
    ]

    prev_state = "state_init"
    kept_states = []
    for t in range(int(window_size)):
        initializers.append(onnx.numpy_helper.from_array(np.asarray(t, dtype=np.int64), f"time_idx_{t}"))
        nodes.extend(
            [
                onnx.helper.make_node("Gather", ["input", f"time_idx_{t}"], [f"x_t_{t}"], axis=1),
                onnx.helper.make_node("MatMul", [f"x_t_{t}", "W_in_T"], [f"in_proj_{t}"]),
                onnx.helper.make_node("MatMul", [prev_state, "W_T"], [f"rec_proj_{t}"]),
                onnx.helper.make_node("Add", [f"in_proj_{t}", f"rec_proj_{t}"], [f"preact_{t}"]),
                onnx.helper.make_node("Tanh", [f"preact_{t}"], [f"candidate_{t}"]),
                onnx.helper.make_node("Mul", [prev_state, "one_minus_leak"], [f"prev_scaled_{t}"]),
                onnx.helper.make_node("Mul", [f"candidate_{t}", "leak"], [f"candidate_scaled_{t}"]),
                onnx.helper.make_node("Add", [f"prev_scaled_{t}", f"candidate_scaled_{t}"], [f"state_{t}"]),
            ]
        )
        prev_state = f"state_{t}"
        if t >= int(washout):
            kept_states.append(prev_state)

    if not kept_states:
        kept_states = [prev_state]

    last_state = kept_states[-1]
    summary_inputs = [last_state]
    if state_summary in {"last_mean", "last_mean_std"}:
        unsqueezed = []
        for idx, state_name in enumerate(kept_states):
            out_name = f"state_seq_{idx}"
            nodes.append(onnx.helper.make_node("Unsqueeze", [state_name, "axis_0"], [out_name]))
            unsqueezed.append(out_name)
        nodes.append(onnx.helper.make_node("Concat", unsqueezed, ["states_seq"], axis=0))
        nodes.append(onnx.helper.make_node("ReduceMean", ["states_seq"], ["states_mean"], axes=[0], keepdims=0))
        summary_inputs.append("states_mean")

    if state_summary == "last_mean_std":
        nodes.extend(
            [
                onnx.helper.make_node("Sub", ["states_seq", "states_mean"], ["states_centered"]),
                onnx.helper.make_node("Mul", ["states_centered", "states_centered"], ["states_var_terms"]),
                onnx.helper.make_node("ReduceMean", ["states_var_terms"], ["states_var"], axes=[0], keepdims=0),
                onnx.helper.make_node("Sqrt", ["states_var"], ["states_std"]),
            ]
        )
        summary_inputs.append("states_std")
    elif state_summary != "last" and state_summary != "last_mean":
        raise ValueError(f"state_summary inconnu: {state_summary}")

    if len(summary_inputs) == 1:
        nodes.append(onnx.helper.make_node("Identity", [summary_inputs[0]], ["summary"]))
    else:
        nodes.append(onnx.helper.make_node("Concat", summary_inputs, ["summary"], axis=1))

    nodes.extend(
        [
            onnx.helper.make_node("Gemm", ["summary", "Wout", "bias"], ["logits"], alpha=1.0, beta=1.0),
            onnx.helper.make_node("Softmax", ["logits"], ["probability_tensor"], axis=1),
            onnx.helper.make_node("ArgMax", ["logits"], ["label_tensor"], axis=1, keepdims=0),
        ]
    )

    graph = onnx.helper.make_graph(
        nodes,
        model_name,
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None, window_size, n_features])],
        [
            onnx.helper.make_tensor_value_info("probability_tensor", onnx.TensorProto.FLOAT, [None, num_classes]),
            onnx.helper.make_tensor_value_info("label_tensor", onnx.TensorProto.INT64, [None]),
        ],
        initializers,
    )
    model = onnx.helper.make_model(
        graph,
        producer_name=f"{MODEL_NAME}_manual_export",
        opset_imports=[onnx.helper.make_operatorsetid("", 15)],
    )
    onnx.checker.check_model(model)
    return model


def train_global_model(df_labeled, df_unlabeled, best_params):
    print("\n" + "=" * 60 + f"\nMODELE GLOBAL - {MODEL_NAME}\n" + "=" * 60)
    config = WINDOW_CONFIGS["default"]
    window_size = config["window_size"]
    step_size = config["step_size"]
    model = None

    try:
        df_all = df_labeled.copy()
        scaler = RobustScaler()
        df_all[SIGNAL_COLS] = scaler.fit_transform(df_all[SIGNAL_COLS])

        # SMOTE sur l'ensemble complet avant l'extraction de features ESN.
        df_all = resample_dataframe(df_all, SIGNAL_COLS)

        reservoir = _build_reservoir(best_params, len(SIGNAL_COLS))
        x_all, y_all = extract_esn_windows(df_all, reservoir, window_size, step_size)
        print(f"  Fenetres labellisees: {len(x_all)}")
        if len(y_all) == 0:
            print("  Aucun exemple global disponible.")
            return

        idx = np.random.default_rng(42).permutation(len(x_all))
        split = int(0.9 * len(x_all))
        x_tr, y_tr = x_all[idx[:split]], y_all[idx[:split]]
        x_vl, y_vl = x_all[idx[split:]], y_all[idx[split:]]

        model = build_model(best_params)
        fit_readout(model, x_tr, y_tr)

        df_unlabeled_fit = df_unlabeled.copy()
        if len(df_unlabeled_fit) > 0:
            df_unlabeled_fit[SIGNAL_COLS] = scaler.transform(df_unlabeled_fit[SIGNAL_COLS])
            x_unlabeled, _ = extract_esn_windows(df_unlabeled_fit, reservoir, window_size, step_size)
            if len(x_unlabeled) > 0 and x_tr.shape[1] == x_unlabeled.shape[1]:
                proba_wrapper = _ReadoutProbaWrapper(model)
                x_tr_v2, y_tr_v2, pseudo_y, _ = add_pseudo_labels(
                    proba_wrapper,
                    x_tr,
                    y_tr,
                    x_unlabeled,
                )
                if len(pseudo_y) > 0:
                    _free_memory(model)
                    model = build_model(best_params)
                    fit_readout(model, x_tr_v2, y_tr_v2)

        models_dir = MODELS_DIR / "ESN"
        models_dir.mkdir(parents=True, exist_ok=True)

        edge_metrics = {}
        
        pkl_path = None
        try:
            pkl_path = models_dir / f"{MODEL_NAME}_global.pkl"
            joblib.dump({"reservoir": reservoir, "readout": model}, pkl_path)
            print(f"  Pickle backup (reservoir + readout) : {pkl_path}")
        except Exception as exc:
            print(f"  WARNING: backup pickle ESN ignore ({type(exc).__name__}: {exc})")
            pkl_path = None

        if len(y_vl) > 0:
            y_pred_native = predict_readout(model, x_vl)
            edge_metrics["native_esn"] = _compute_classification_metrics(y_vl, y_pred_native)
            if pkl_path is not None and pkl_path.exists():
                edge_metrics["native_esn"]["model_size_kb"] = round(pkl_path.stat().st_size / 1024, 2)
        else:
            edge_metrics["native_esn"] = {}

        try:
            W_in, W, leak_rate = _extract_reservoir_weights(reservoir)
            onnx_model = _build_onnx_esn_graph(
                W_in, W, leak_rate,
                model.Wout, model.bias,
                window_size, len(SIGNAL_COLS), int(best_params["n_reservoir"]),
                len(LABEL_MAPPING), int(best_params.get("washout", 0)),
                best_params.get("state_summary", "last_mean_std"),
                f"{MODEL_NAME}_global"
            )

            raw_sample = _extract_raw_windows_sample(df_all, window_size, step_size, n_samples=20)
            if len(raw_sample) > 0:
                _validate_onnx_esn(onnx_model, reservoir, model, raw_sample)

            onnx_path = models_dir / f"{MODEL_NAME}_global.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"  ONNX exporté : {onnx_path}")

            _export_c_array_from_bytes(onnx_path, models_dir, f"{MODEL_NAME}_global")

            if len(x_vl) > 0:
                y_pred_onnx = _predict_onnx(onnx_path, x_vl)
                edge_metrics["onnx"] = _compute_classification_metrics(y_vl, y_pred_onnx)
                edge_metrics["onnx"]["model_size_kb"] = round(onnx_path.stat().st_size / 1024, 2)

        except Exception as exc:
            print(f"  Export ONNX échoué ({type(exc).__name__}: {exc})")
            traceback.print_exc()

        _save_edge_comparison_metrics(MODEL_NAME, edge_metrics)
        print("  Modele global ESN traite (native + export ONNX).")

    except Exception as exc:
        print(f"  Modele global echoue ({type(exc).__name__}: {exc})")
        traceback.print_exc()
    finally:
        _free_memory(model)
        gc.collect()


# ============================================================================
# 7. BOUCLE LOSO
# ============================================================================
def main():
    """
    Fonction principale : exécute le pipeline complet LOSO (Leave-One-Session-Out).
    Gère le chargement des données, l'optimisation, l'entraînement par fold et la sauvegarde des métriques.
    """
    print("\n" + "=" * 60 + f"\nTRAIN LOSO - {MODEL_NAME}\n" + "=" * 60)
    if not DATA_MODEL_READY.exists():
        return print(f"Dataset non trouve: {DATA_MODEL_READY}")

    # Chargement et mapping des labels.
    df = pd.read_csv(DATA_MODEL_READY)
    df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors="coerce").fillna(-1).astype(int)
    df_labeled = df[df[COL_LABEL] >= 0].copy()
    df_unlabeled = df[df[COL_LABEL] == -1].copy()
    print(f"Labeled: {len(df_labeled)} lignes | Unlabeled: {len(df_unlabeled)} lignes")

    # Liste de toutes les sessions uniques (Participant, Session).
    unique_sessions = [
        tuple(x)
        for x in df_labeled[df_labeled[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]]
        .drop_duplicates()
        .values
    ]

    # prendre les paramètres optimisés s’ils existent, mais garder les valeurs par défaut pour ceux qui manquent.
    best_params = optimize_hyperparams(df_labeled) if USE_OPTUNA_ESN else MODEL_PARAMS["ESN"]
    best_params = {**MODEL_PARAMS["ESN"], **best_params}

    all_metrics = []
    plots_dir = BASE_DIR / "training_curves" / MODEL_NAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Boucle LOSO : on teste chaque session une par une.
    for fold_idx, (test_part, test_sess) in enumerate(unique_sessions, start=1):
        print(f"\n--- FOLD {fold_idx}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")
        config = WINDOW_CONFIGS.get(test_part, WINDOW_CONFIGS["default"])
        window_size = config["window_size"]
        step_size = config["step_size"]

        # Train = tout sauf la session test actuelle.
        df_train = df_labeled[~((df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess))].copy()
        df_test = df_labeled[(df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess)].copy()

        model = None
        try:
            # SMOTE seulement sur df_train, avant le scaling et l'extraction de features ESN.
            df_train = resample_dataframe(df_train, SIGNAL_COLS)

            # Scaling spécifique au fold.
            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])
            df_unlabeled_fit = df_unlabeled.copy()
            if len(df_unlabeled_fit) > 0:
                df_unlabeled_fit[SIGNAL_COLS] = scaler.transform(df_unlabeled_fit[SIGNAL_COLS])

            # Extraction features ESN.
            reservoir = _build_reservoir(best_params, len(SIGNAL_COLS))
            x_train, y_train = extract_esn_windows(df_train, reservoir, window_size, step_size)
            x_test, y_test = extract_esn_windows(df_test, reservoir, window_size, step_size)

            if len(y_train) == 0 or len(y_test) == 0:
                print("  WARNING: train/test vide. Fold ignore.")
                continue

            print(f"  Train: {len(x_train)} fenetres | Test: {len(x_test)} fenetres")

            # Entraînement du readout Ridge.
            model = build_model(best_params)
            fit_readout(model, x_train, y_train)
            x_unlabeled, _ = (
                extract_esn_windows(df_unlabeled_fit, reservoir, window_size, step_size)
                if len(df_unlabeled_fit) > 0
                else (np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int32))
            )

            if len(x_unlabeled) > 0 and x_train.shape[1] == x_unlabeled.shape[1]:
                proba_wrapper = _ReadoutProbaWrapper(model)
                x_train_v2, y_train_v2, pseudo_y, _ = add_pseudo_labels(
                    proba_wrapper,
                    x_train,
                    y_train,
                    x_unlabeled,
                )
                if len(pseudo_y) > 0:
                    _free_memory(model)
                    model = build_model(best_params)
                    fit_readout(model, x_train_v2, y_train_v2)

            y_pred = predict_readout(model, x_test)

            # Calcul des métriques.
            f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            # Rapport de classification détaillé.
            labels_present = np.unique(np.concatenate([y_test, y_pred]))
            print(
                "\n"
                + classification_report(
                    y_test,
                    y_pred,
                    labels=labels_present,
                    target_names=[TARGET_NAMES[i] for i in labels_present],
                    zero_division=0,
                )
            )

            # Diagnostic visuel.
            states = first_window_states(df_test, reservoir, window_size)
            plot_fold(
                fold_idx,
                test_part,
                test_sess,
                plots_dir,
                f1_mac,
                bal_acc,
                y_test,
                y_pred,
                states,
            )

            all_metrics.append(
                {
                    "Fold": fold_idx,
                    "Participant": int(test_part),
                    "Session": int(test_sess),
                    "F1_Macro": float(f1_mac),
                    "Balanced_Accuracy": float(bal_acc),
                }
            )

        except Exception as exc:
            print(f"  Fold {fold_idx} echoue ({type(exc).__name__}: {exc})")
        finally:
            _free_memory(model)

    if not all_metrics:
        return print("Aucun fold complete.")

    # Synthèse des résultats.
    df_res = pd.DataFrame(all_metrics)
    print("\n" + "=" * 60 + f"\nRESULTATS FINAUX - {MODEL_NAME}\n" + "=" * 60)
    print(df_res.describe().loc[["mean", "std"]])

    # Mise à jour du fichier metrics.json central.
    curr_metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                content = f.read().strip()
            curr_metrics = json.loads(content) if content else {}
        except (json.JSONDecodeError, ValueError):
            print(f"metrics.json invalide ou vide, reinitialisation: {METRICS_PATH}")
            curr_metrics = {}

    curr_metrics[MODEL_NAME] = {
        "mean_f1": float(df_res["F1_Macro"].mean()),
        "std_f1": float(df_res["F1_Macro"].std()),
        "mean_bal_acc": float(df_res["Balanced_Accuracy"].mean()),
        "std_bal_acc": float(df_res["Balanced_Accuracy"].std()),
        "params": best_params,
        "folds": all_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(curr_metrics, f, indent=4)
    print(f"\nMetriques -> {METRICS_PATH}")

    # Entraînement final global.
    train_global_model(df_labeled, df_unlabeled, best_params)


if __name__ == "__main__":
    main()
