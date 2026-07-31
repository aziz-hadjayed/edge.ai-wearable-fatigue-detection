import gc
import inspect
import importlib.util
import json
import random
import sys
import traceback
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnxruntime as ort
import optuna
import pandas as pd
from reservoirpy.nodes import RLS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import *
from models.semi_supervised import add_pseudo_labels
from utils.apply_smote import resample_dataframe

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

_BASE_SPEC = importlib.util.spec_from_file_location(
    "train_esn_base",
    Path(__file__).resolve().with_name("train_esn.py"),
)
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


MODEL_NAME = "ESN_RLS"
LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES = ["baseline", "activity", "pre_fatigue", "fatigue"]
RLS_FEATURE_CLIP = 8.0


# ============================================================================
# 1. OUTILS GENERAUX
# ============================================================================
def _free_memory(*objects):
    """
    Libere explicitement les objets temporaires et force le garbage collector.
    Utile pendant les folds LOSO et les essais Optuna pour limiter l'usage RAM.
    """
    for obj in objects:
        del obj
    gc.collect()


# ============================================================================
# 2. READOUT RLS ET PRETRAITEMENT DES FEATURES
# ============================================================================
def build_model(params):
    """
    Cree le readout RLS qui apprend a classifier les features produites par l'ESN.
    Le Reservoir est construit dans train_esn.py et seul le readout change ici.
    """
    return RLS(
        alpha=float(params["rls_alpha"]),
        forgetting=float(params.get("forgetting", 1.0)),
        output_dim=len(LABEL_MAPPING),
        name="readout_rls",
    )


def _to_onehot(y, num_classes=None):
    """
    Encode les labels numeriques en vecteurs One-Hot pour l'apprentissage du RLS.
    """
    num_classes = num_classes or len(LABEL_MAPPING)
    y_onehot = np.zeros((len(y), num_classes), dtype=np.float32)
    y_onehot[np.arange(len(y)), y.astype(int)] = 1.0
    return y_onehot


def _clean_features(x):
    """
    Nettoie les features ESN avant le RLS : remplace NaN/inf et limite les valeurs extremes.
    Cette protection evite les instabilites numeriques du readout recursif.
    """
    x = np.nan_to_num(x, nan=0.0, posinf=RLS_FEATURE_CLIP, neginf=-RLS_FEATURE_CLIP)
    return np.clip(x, -RLS_FEATURE_CLIP, RLS_FEATURE_CLIP).astype(np.float32)


def fit_feature_scaler(x_train):
    """
    Ajuste un StandardScaler sur les features d'entrainement et renvoie les features nettoyees.
    Le scaling se fait apres l'extraction ESN, donc sur les etats resumes du reservoir.
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    return scaler, _clean_features(x_scaled)


def transform_features(feature_scaler, x):
    """
    Applique le scaler deja appris aux features de validation/test puis les nettoie.
    """
    return _clean_features(feature_scaler.transform(x))


def fit_readout(model, x_train, y_train):
    """
    Entraine le readout RLS sur les features ESN.
    Les classes sont ponderees pour compenser le desequilibre du dataset.
    """
    # La racine carree des poids permet d'appliquer la ponderation sur X et y.
    weights = np.sqrt(base._sample_weights(y_train)).reshape(-1, 1)
    x_train = _clean_features(x_train)
    x_weighted = x_train * weights
    y_weighted = _to_onehot(y_train) * weights

    # Les RuntimeWarning sont transformes en erreurs pour detecter les divergences RLS.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        model.fit(x_weighted, y_weighted)

    # Verification finale : un poids non fini rendrait toutes les predictions invalides.
    if not np.all(np.isfinite(model.Wout)):
        raise FloatingPointError("RLS Wout contient des valeurs non finies")
    if not np.all(np.isfinite(model.bias)):
        raise FloatingPointError("RLS bias contient des valeurs non finies")
    return model


def predict_readout(model, x):
    """
    Calcule les scores du RLS puis retourne la classe avec le score maximal.
    """
    scores = np.asarray(model.run(_clean_features(x)), dtype=np.float32)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("RLS predictions contiennent des valeurs non finies")
    # Argmax sur les sorties One-Hot apprises : 0=baseline, 1=activity, 2=pre_fatigue, 3=fatigue.
    return np.argmax(scores, axis=1).astype(np.int32)


class _ReadoutProbaWrapper:
    """
    Adapte le RLS readout pour exposer predict_proba, compatible avec add_pseudo_labels().
    """
    def __init__(self, readout_model):
        self.readout_model = readout_model

    def predict_proba(self, x):
        scores = np.asarray(self.readout_model.run(_clean_features(x)), dtype=np.float32)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


# ============================================================================
# 3. OPTUNA
# ============================================================================
def _suggest_esn_rls_params(trial):
    """
    Definit l'espace de recherche Optuna pour le couple ESN + RLS.
    Les hyperparametres ESN controlent le reservoir, ceux du RLS controlent le readout.
    """
    space = ESN_RLS_OPTUNA_SPACE
    return {
        "n_reservoir": trial.suggest_categorical("n_reservoir", space["n_reservoir"]),
        "spectral_radius": trial.suggest_float("spectral_radius", **space["spectral_radius"]),
        "sparsity": trial.suggest_float("sparsity", **space["sparsity"]),
        "leak_rate": trial.suggest_float("leak_rate", **space["leak_rate"]),
        "input_scaling": trial.suggest_float("input_scaling", **space["input_scaling"]),
        "rls_alpha": trial.suggest_float("rls_alpha", **space["rls_alpha"]),
        "forgetting": trial.suggest_float("forgetting", **space["forgetting"]),
        "washout": trial.suggest_categorical("washout", space["washout"]),
        "state_summary": trial.suggest_categorical("state_summary", space["state_summary"]),
        "random_state": 42,
    }


def optuna_objective(trial, df, sessions, window_size, step_size):
    """
    Evalue un jeu d'hyperparametres Optuna sur quelques sessions de validation.
    Chaque session est retiree de l'entrainement puis utilisee comme validation temporaire.
    """
    params = _suggest_esn_rls_params(trial)
    scores = []

    for val_part, val_sess in sessions:
        model = None
        try:
            # Split session-wise : la session de validation ne fuit pas dans l'entrainement.
            df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
            df_val = df[(df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess)].copy()

            # SMOTE seulement sur df_train, avant le scaling et l'extraction de features ESN.
            df_train = resample_dataframe(df_train, SIGNAL_COLS)

            # Normalisation des signaux bruts avant passage dans le reservoir ESN.
            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_val[SIGNAL_COLS] = scaler.transform(df_val[SIGNAL_COLS])

            # Extraction des features ESN avec la logique commune du script train_esn.py.
            reservoir = base._build_reservoir(params, len(SIGNAL_COLS))
            x_train, y_train = base.extract_esn_windows(df_train, reservoir, window_size, step_size)
            x_val, y_val = base.extract_esn_windows(df_val, reservoir, window_size, step_size)
            if len(y_train) == 0 or len(y_val) == 0:
                scores.append(0.0)
                continue
            # Deuxieme scaling : il stabilise les etats du reservoir avant le RLS.
            feature_scaler, x_train = fit_feature_scaler(x_train)
            x_val = transform_features(feature_scaler, x_val)

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


def optimize_hyperparams(df, n_trials=ESN_RLS_OPTUNA_TRIALS):
    """
    Lance l'etude Optuna et sauvegarde les meilleurs hyperparametres ESN_RLS.
    """
    print(f"\nOPTUNA ESN_RLS - {n_trials} trials | {ESN_RLS_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
    random.seed(42)
    window_size = WINDOW_CONFIGS["default"]["window_size"]
    step_size = WINDOW_CONFIGS["default"]["step_size"]
    # Liste des sessions disponibles pour construire un petit set de validation Optuna.
    all_sessions = [
        tuple(map(int, x))
        for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
    ]
    val_sessions = random.sample(all_sessions, min(ESN_RLS_OPTUNA_SESSIONS, len(all_sessions)))
    print(f"Sessions Optuna: {val_sessions}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(
        # Chaque trial mesure le F1 macro moyen sur les sessions de validation choisies.
        lambda trial: optuna_objective(trial, df, val_sessions, window_size, step_size),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
        gc_after_trial=True,
        catch=(Exception,),
    )

    print(f"\nBest F1-Macro: {study.best_value:.4f}")
    print(f"Best params  : {study.best_params}")
    # Persistance des resultats pour reutiliser les meilleurs parametres sans relancer Optuna.
    OPTUNA_PATH_ESN_RLS.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_ESN_RLS, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return {**MODEL_PARAMS["ESN_RLS"], **study.best_params}


# ============================================================================
# 4. VISUALISATION
# ============================================================================
def plot_fold(fold_idx, test_part, test_sess, save_dir, f1_mac, bal_acc, y_true, y_pred, states):
    """
    Genere les graphiques d'un fold : scores, matrice de confusion et energie du reservoir.
    """
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        f"ESN_RLS | Fold {fold_idx} - P{test_part} S{test_sess}"
        f" | F1-Macro: {f1_mac:.3f} | Bal.Acc: {bal_acc:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    # Resume visuel des deux metriques principales du fold.
    axes[0].bar(["F1-Macro", "Balanced Acc"], [f1_mac, bal_acc], color=["#2563EB", "#059669"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Scores du fold")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Matrice de confusion limitee aux classes presentes dans y_true/y_pred.
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

    if states is not None and len(states) > 0:
        # Energie moyenne des etats internes : utile pour voir si le reservoir reste stable.
        energy = np.mean(np.abs(states), axis=1)
        axes[2].plot(np.arange(len(energy)), energy, color="#EA580C", linewidth=2)
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
# 5. MODELE GLOBAL
# ============================================================================
def _build_onnx_esn_rls_graph(
    W_in,
    W,
    leak_rate,
    mean,
    scale,
    clip_value,
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
    W_in_T = np.asarray(W_in, dtype=np.float32).T
    W_T = np.asarray(W, dtype=np.float32).T
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    scale = np.asarray(scale, dtype=np.float32).reshape(1, -1)
    Wout = np.asarray(Wout, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32).reshape(-1)

    feature_dim = int(mean.shape[1])
    nodes = []
    initializers = [
        onnx.numpy_helper.from_array(W_in_T, "W_in_T"),
        onnx.numpy_helper.from_array(W_T, "W_T"),
        onnx.numpy_helper.from_array(mean, "feature_mean"),
        onnx.numpy_helper.from_array(scale, "feature_scale"),
        onnx.numpy_helper.from_array(np.asarray(Wout, dtype=np.float32), "Wout"),
        onnx.numpy_helper.from_array(bias.astype(np.float32), "bias"),
        onnx.numpy_helper.from_array(np.asarray([1.0 - float(leak_rate)], dtype=np.float32), "one_minus_leak"),
        onnx.numpy_helper.from_array(np.asarray([float(leak_rate)], dtype=np.float32), "leak"),
        onnx.numpy_helper.from_array(np.asarray([-float(clip_value)], dtype=np.float32), "clip_min"),
        onnx.numpy_helper.from_array(np.asarray([float(clip_value)], dtype=np.float32), "clip_max"),
        onnx.numpy_helper.from_array(np.zeros((1, n_reservoir), dtype=np.float32), "state_init"),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "axis_0"),
    ]

    prev_state = "state_init"
    kept_states = []
    for t in range(int(window_size)):
        initializers.append(onnx.numpy_helper.from_array(np.asarray(t, dtype=np.int64), f"time_idx_{t}"))
        nodes.extend(
            [
                onnx.helper.make_node(
                    "Gather",
                    ["input", f"time_idx_{t}"],
                    [f"x_t_{t}"],
                    axis=1,
                ),
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
            onnx.helper.make_node("Sub", ["summary", "feature_mean"], ["summary_centered"]),
            onnx.helper.make_node("Div", ["summary_centered", "feature_scale"], ["summary_scaled_raw"]),
            onnx.helper.make_node("Clip", ["summary_scaled_raw", "clip_min", "clip_max"], ["summary_scaled"]),
            onnx.helper.make_node("Gemm", ["summary_scaled", "Wout", "bias"], ["logits"], alpha=1.0, beta=1.0),
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
    if feature_dim != Wout.shape[0]:
        raise ValueError(f"Dimension feature scaler/readout incoherente: {feature_dim} != {Wout.shape[0]}")
    return model


def _extract_raw_windows_sample(df, window_size, step_size, n_samples=None):
    """
    Extrait des fenetres BRUTES (avant reservoir), shape (n, window_size, n_features).
    A la difference de base.extract_esn_windows qui retourne les features deja resumees
    par le reservoir, cette fonction retourne les fenetres telles quelles, necessaires
    pour alimenter le graphe ONNX (qui embarque lui-meme la recurrence) et pour valider
    numeriquement ce graphe face au chemin reservoirpy reel.
    Retourne aussi les labels correspondants (meme logique de vote majoritaire que
    base._window_label), pour permettre le calcul de edge_metrics sur les memes echantillons.
    """
    df = base._sort_df(df)
    raw_windows, raw_labels = [], []
    for _, group in df.groupby([COL_PARTICIPANT, COL_SESSION], sort=False):
        x_raw = group[SIGNAL_COLS].values.astype(np.float32)
        y_raw = group[COL_LABEL].values
        for start in range(0, len(x_raw) - window_size + 1, step_size):
            end = start + window_size
            y_window = y_raw[start:end]
            if np.isnan(y_window).any():
                continue
            raw_windows.append(x_raw[start:end])
            raw_labels.append(base._window_label(y_window.astype(np.int32)))
            if n_samples is not None and len(raw_windows) >= n_samples:
                return (
                    np.asarray(raw_windows, dtype=np.float32),
                    np.asarray(raw_labels, dtype=np.int32),
                )
    if not raw_windows:
        return (
            np.empty((0, window_size, len(SIGNAL_COLS)), dtype=np.float32),
            np.array([], dtype=np.int32),
        )
    return np.asarray(raw_windows, dtype=np.float32), np.asarray(raw_labels, dtype=np.int32)


def _missing_base_helpers():
    required = [
        "_extract_reservoir_weights",
        "_validate_onnx_esn",
        "_predict_onnx",
        "_export_c_array_from_bytes",
        "_compute_classification_metrics",
        "_save_edge_comparison_metrics",
    ]
    return [name for name in required if not hasattr(base, name)]


def _missing_base_helpers_for(names):
    return [name for name in names if not hasattr(base, name)]


def _validate_base_metric_helper():
    expected_keys = {
        "f1_macro",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "precision_fatigue",
        "recall_fatigue",
    }
    func = base._compute_classification_metrics
    signature = inspect.signature(func)
    if list(signature.parameters) != ["y_true", "y_pred"]:
        raise TypeError(
            "base._compute_classification_metrics doit avoir exactement la signature "
            "(y_true, y_pred)"
        )

    sample_true = np.array([0, 1, 2, 3], dtype=np.int32)
    sample_pred = np.array([0, 1, 2, 3], dtype=np.int32)
    result = func(sample_true, sample_pred)
    if not isinstance(result, dict):
        raise TypeError("base._compute_classification_metrics doit retourner un dict")
    actual_keys = set(result.keys())
    if actual_keys != expected_keys:
        raise ValueError(
            "base._compute_classification_metrics doit retourner exactement les clés "
            f"{sorted(expected_keys)}; reçu {sorted(actual_keys)}"
        )


def train_global_model(df_labeled, df_unlabeled, best_params):
    print("\n" + "=" * 60 + f"\nMODELE GLOBAL - {MODEL_NAME}\n" + "=" * 60)

    missing = _missing_base_helpers()
    if missing:
        print(f"  Modele global echoue : fonctions manquantes dans train_esn.py : {missing}")
        return
    try:
        _validate_base_metric_helper()
    except (TypeError, ValueError) as exc:
        print(f"  Modele global echoue : {exc}")
        return

    config = WINDOW_CONFIGS["default"]
    window_size, step_size = config["window_size"], config["step_size"]

    df_all = df_labeled.copy()
    scaler = RobustScaler()
    df_all[SIGNAL_COLS] = scaler.fit_transform(df_all[SIGNAL_COLS])
    df_all = resample_dataframe(df_all, SIGNAL_COLS)

    reservoir = base._build_reservoir(best_params, len(SIGNAL_COLS))
    X_all, y_all = base.extract_esn_windows(df_all, reservoir, window_size, step_size)
    print(f"  Fenêtres labellisées : {len(X_all)}")

    model = None
    try:
        idx = np.random.default_rng(42).permutation(len(X_all))
        split = int(0.9 * len(X_all))
        X_tr_raw, y_tr = X_all[idx[:split]], y_all[idx[:split]]
        X_vl_raw, y_vl = X_all[idx[split:]], y_all[idx[split:]]

        feature_scaler, X_tr = fit_feature_scaler(X_tr_raw)
        X_vl = transform_features(feature_scaler, X_vl_raw)

        model = build_model(best_params)
        fit_readout(model, X_tr, y_tr)

        # Pseudo-labeling
        df_unl = df_unlabeled.copy()
        if len(df_unl) > 0:
            df_unl[SIGNAL_COLS] = scaler.transform(df_unl[SIGNAL_COLS])
            X_unlabeled_raw, _ = base.extract_esn_windows(df_unl, reservoir, window_size, step_size)
            if len(X_unlabeled_raw) > 0 and X_tr.shape[1] == X_unlabeled_raw.shape[1]:
                X_unlabeled = transform_features(feature_scaler, X_unlabeled_raw)
                proba_wrapper = _ReadoutProbaWrapper(model)
                X_tr_v2, y_tr_v2, pseudo_y, _ = add_pseudo_labels(proba_wrapper, X_tr, y_tr, X_unlabeled)
                if len(pseudo_y) > 0:
                    _free_memory(model)
                    model = build_model(best_params)
                    fit_readout(model, X_tr_v2, y_tr_v2)

        models_dir = MODELS_DIR / "ESN_RLS"
        models_dir.mkdir(parents=True, exist_ok=True)

        edge_metrics = {}

        # Backup natif
        pkl_path = None
        try:
            pkl_path = models_dir / f"{MODEL_NAME}_global.pkl"
            joblib.dump({"reservoir": reservoir, "readout": model, "feature_scaler": feature_scaler}, pkl_path)
            print(f"  Pickle backup : {pkl_path}")
        except Exception as exc:
            print(f"  Pickle backup échoué ({type(exc).__name__}: {exc})")
            pkl_path = None

        y_pred_native = predict_readout(model, X_vl_raw)
        edge_metrics["native_esn_rls"] = base._compute_classification_metrics(y_vl, y_pred_native)
        if pkl_path is not None and pkl_path.exists():
            edge_metrics["native_esn_rls"]["model_size_kb"] = round(pkl_path.stat().st_size / 1024, 2)

        # Export ONNX
        try:
            W_in, W, leak_rate = base._extract_reservoir_weights(reservoir)
            print(f"  Wout.shape (reservoirpy RLS) : {model.Wout.shape}")
            onnx_model = _build_onnx_esn_rls_graph(
                W_in, W, leak_rate,
                feature_scaler.mean_, feature_scaler.scale_, RLS_FEATURE_CLIP,
                model.Wout, model.bias,
                window_size, len(SIGNAL_COLS), int(best_params["n_reservoir"]),
                len(LABEL_MAPPING), int(best_params.get("washout", 0)),
                best_params.get("state_summary", "last_mean_std"),
                f"{MODEL_NAME}_global"
            )

            # Extraire des fenetres brutes (avant reservoir) pour la validation et l'evaluation ONNX,
            # car le graphe ONNX attend l'entree brute, pas les features deja resumees (X_vl_raw)
            raw_windows_sample, raw_labels_sample = _extract_raw_windows_sample(
                df_all, window_size, step_size, n_samples=20
            )
            if len(raw_windows_sample) > 0:
                base._validate_onnx_esn(onnx_model, reservoir, model, raw_windows_sample)

            onnx_path = models_dir / f"{MODEL_NAME}_global.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"  ONNX exporté : {onnx_path}")

            base._export_c_array_from_bytes(onnx_path, models_dir, f"{MODEL_NAME}_global")

            raw_windows_eval, raw_labels_eval = _extract_raw_windows_sample(
                df_all, window_size, step_size, n_samples=None
            )
            if len(raw_windows_eval) > 0:
                y_pred_onnx = base._predict_onnx(onnx_path, raw_windows_eval)
                # Limitation : edge_metrics["onnx"] compare aux labels extraits des fenetres brutes
                # (raw_labels_eval), pas au split validation 90/10 (y_vl) utilise pour native_esn_rls.
                # La comparaison native vs onnx n'est donc pas parfaitement appariee echantillon par echantillon.
                edge_metrics["onnx"] = base._compute_classification_metrics(raw_labels_eval, y_pred_onnx)
                edge_metrics["onnx"]["model_size_kb"] = round(onnx_path.stat().st_size / 1024, 2)

        except Exception as exc:
            print(f"  Export ONNX échoué ({type(exc).__name__}: {exc})")
            traceback.print_exc()

        base._save_edge_comparison_metrics(MODEL_NAME, edge_metrics)
        print(f"  Modèle global ESN_RLS traité (native + export ONNX si réussi).")

    except Exception as exc:
        print(f"  Modèle global échoué ({type(exc).__name__}: {exc})")
        traceback.print_exc()
    finally:
        _free_memory(model)
        gc.collect()


# ============================================================================
# 6. ENTRAINEMENT LOSO
# ============================================================================
def main():
    """
    Point d'entree du script : charge les donnees, entraine en LOSO, sauvegarde les metriques.
    LOSO signifie que chaque fold garde une session participant comme test.
    """
    print("\n" + "=" * 60 + f"\nTRAIN LOSO - {MODEL_NAME}\n" + "=" * 60)
    if not DATA_MODEL_READY.exists():
        return print(f"Dataset non trouve: {DATA_MODEL_READY}")

    df = pd.read_csv(DATA_MODEL_READY)
    df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors="coerce").fillna(-1).astype(int)
    df_labeled = df[df[COL_LABEL] >= 0].copy()
    df_unlabeled = df[df[COL_LABEL] == -1].copy()
    print(f"Labeled: {len(df_labeled)} lignes | Unlabeled: {len(df_unlabeled)} lignes")

    # Chaque couple participant/session devient un fold de test.
    unique_sessions = [
        tuple(map(int, x))
        for x in df_labeled[df_labeled[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]]
        .drop_duplicates()
        .values
    ]

    best_params = optimize_hyperparams(df_labeled) if USE_OPTUNA_ESN_RLS else MODEL_PARAMS["ESN_RLS"]
    best_params = {**MODEL_PARAMS["ESN_RLS"], **best_params}

    all_metrics = []
    plots_dir = BASE_DIR / "training_curves" / MODEL_NAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (test_part, test_sess) in enumerate(unique_sessions, start=1):
        print(f"\n--- FOLD {fold_idx}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")
        config = WINDOW_CONFIGS.get(test_part, WINDOW_CONFIGS["default"])
        window_size = config["window_size"]
        step_size = config["step_size"]

        # Split LOSO : la session test est completement exclue de l'entrainement.
        df_train = df_labeled[~((df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess))].copy()
        df_test = df_labeled[(df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess)].copy()

        # SMOTE seulement sur df_train, avant le scaling et l'extraction de features ESN.
        df_train = resample_dataframe(df_train, SIGNAL_COLS)

        model = None
        try:
            # RobustScaler limite l'effet des valeurs extremes dans les signaux capteurs.
            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])

            df_unlabeled_fit = df_unlabeled.copy()
            if len(df_unlabeled_fit) > 0:
                df_unlabeled_fit[SIGNAL_COLS] = scaler.transform(df_unlabeled_fit[SIGNAL_COLS])

            # Le reservoir extrait une representation temporelle fixe pour chaque fenetre.
            reservoir = base._build_reservoir(best_params, len(SIGNAL_COLS))
            x_train, y_train = base.extract_esn_windows(df_train, reservoir, window_size, step_size)
            x_test, y_test = base.extract_esn_windows(df_test, reservoir, window_size, step_size)

            if len(y_train) == 0 or len(y_test) == 0:
                print("  WARNING: train/test vide. Fold ignore.")
                continue
            # Normalisation des features ESN avant apprentissage/prediction RLS.
            feature_scaler, x_train = fit_feature_scaler(x_train)
            x_test = transform_features(feature_scaler, x_test)

            print(f"  Train: {len(x_train)} fenetres | Test: {len(x_test)} fenetres")

            model = build_model(best_params)
            fit_readout(model, x_train, y_train)

            x_unlabeled, _ = (
                base.extract_esn_windows(df_unlabeled_fit, reservoir, window_size, step_size)
                if len(df_unlabeled_fit) > 0 else (np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int32))
            )

            if len(x_unlabeled) > 0 and x_train.shape[1] == x_unlabeled.shape[1]:
                x_unlabeled = transform_features(feature_scaler, x_unlabeled)
                proba_wrapper = _ReadoutProbaWrapper(model)
                x_train_v2, y_train_v2, pseudo_y, _ = add_pseudo_labels(proba_wrapper, x_train, y_train, x_unlabeled)
                if len(pseudo_y) > 0:
                    _free_memory(model)
                    model = build_model(best_params)
                    fit_readout(model, x_train_v2, y_train_v2)

            y_pred = predict_readout(model, x_test)

            f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

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

            # Etats de la premiere fenetre test pour visualiser la dynamique du reservoir.
            states = base.first_window_states(df_test, reservoir, window_size)
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

    # Synthese des metriques sur tous les folds LOSO.
    df_res = pd.DataFrame(all_metrics)
    print("\n" + "=" * 60 + f"\nRESULTATS FINAUX - {MODEL_NAME}\n" + "=" * 60)
    print(df_res.describe().loc[["mean", "std"]])

    # Lecture defensive du metrics.json existant : s'il est vide/invalide, on repart proprement.
    curr_metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                content = f.read().strip()
            curr_metrics = json.loads(content) if content else {}
        except (json.JSONDecodeError, ValueError):
            print(f"metrics.json invalide ou vide, reinitialisation: {METRICS_PATH}")
            curr_metrics = {}

    # Ajout/remplacement des resultats ESN_RLS dans le fichier de metriques global.
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

    # Entrainement final sur toutes les donnees apres l'evaluation LOSO.
    train_global_model(df_labeled, df_unlabeled, best_params)


if __name__ == "__main__":
    main()
"""Train ESN-RLS on the dataset and evaluate it with LOSO cross-validation."""
