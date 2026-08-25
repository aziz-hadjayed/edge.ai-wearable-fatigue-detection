import gc
import json
import random
import sys
import traceback
import warnings
from pathlib import Path
from typing import cast

# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
import optuna
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import *

from utils.apply_smote import resample_dataframe

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 1. CONSTANTES
# ============================================================================
MODEL_NAME = "QKERNEL"
LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES = ["baseline", "activity", "pre_fatigue", "fatigue"]


def _free_memory(*objects):
    for obj in objects:
        del obj
    gc.collect()


# ============================================================================
# 2. CIRCUIT D'ENCODAGE QUANTIQUE - SIMULATEUR STATE-VECTOR NUMPY
# ============================================================================
# Formalisme repris tel quel de QRC.py (state-vector NumPy pur, pas de
# PennyLane) -- seules les primitives effectivement utilisees par un encodeur
# a une seule passe (RY d'encodage + une couche d'intrication CNOT en anneau)
# sont reprises ici.
def _ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex64)


def _apply_single_qubit_gate(state, gate, qubit, n_qubits):
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    tensor = np.tensordot(gate, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(-1)


def _cnot_permutation(n_qubits, control, target):
    dim = 2 ** n_qubits
    perm = np.arange(dim)
    control_mask = 1 << (n_qubits - 1 - control)
    target_mask = 1 << (n_qubits - 1 - target)
    for idx in range(dim):
        if idx & control_mask:
            perm[idx] = idx ^ target_mask
    return perm


def _apply_permutation(state, perm):
    new_state = np.empty_like(state)
    new_state[perm] = state
    return new_state


def _build_encoder(params, input_dim):
    """Circuit fixe : rotation d'encodage (RY par qubit, angle derive des
    features projetees) + UNE SEULE couche d'intrication (CNOT en anneau).
    Pas de recurrence, pas de repetition -- applique une fois par fenetre."""
    n_qubits = int(params["n_qubits"])
    rng = np.random.default_rng(int(params.get("random_state", 42)))
    cnot_pairs = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    cnot_perms = [_cnot_permutation(n_qubits, c, t) for c, t in cnot_pairs]
    projection_weights = rng.normal(0.0, 1.0, size=(input_dim, n_qubits)).astype(np.float32)
    projection_bias = rng.uniform(-np.pi, np.pi, size=n_qubits).astype(np.float32)
    return {
        "params": dict(params),
        "n_qubits": n_qubits,
        "projection_weights": projection_weights,
        "projection_bias": projection_bias,
        "cnot_perms": cnot_perms,
    }


def encode_state(x, encoder):
    """Encode UN vecteur de features (deja scale) en etat quantique.
    Bornage tanh applique DES LE DEPART (lecon tiree de l'investigation QRC :
    l'aliasing d'angle detruit le signal, corrige ici par construction plutot
    qu'apres coup)."""
    params = encoder["params"]
    n_qubits = encoder["n_qubits"]
    input_scaling = float(params["input_scaling"])

    angles_raw = input_scaling * (x @ encoder["projection_weights"]) + encoder["projection_bias"]
    angles = np.pi * np.tanh(angles_raw / np.pi)  # borne dans (-pi, pi), strictement monotone

    state = np.zeros(2 ** n_qubits, dtype=np.complex64)
    state[0] = 1.0 + 0.0j
    for q, angle in enumerate(angles):
        state = _apply_single_qubit_gate(state, _ry(float(angle)), q, n_qubits)
    for perm in encoder["cnot_perms"]:
        state = _apply_permutation(state, perm)

    norm = np.linalg.norm(state)
    if norm > 0:
        state = state / norm
    return state


def encode_batch(X, encoder):
    """Encode toutes les fenetres d'un coup. Retourne une matrice
    (n_samples, 2**n_qubits) complexe -- chaque ligne est un etat quantique."""
    return np.stack([encode_state(x, encoder) for x in X], axis=0)


def compute_kernel_matrix(states_a, states_b):
    """K[i,j] = |<psi_i|psi_j>|^2, vectorise via un seul produit matriciel."""
    overlaps = states_a @ states_b.conj().T   # (n_a, n_b) complexe
    return np.abs(overlaps) ** 2               # (n_a, n_b) reel, dans [0,1]


# ============================================================================
# 3. DONNEES - FEATURES CLASSIQUES PAR FENETRE
# ============================================================================
def _window_label(y_window):
    values, counts = np.unique(y_window, return_counts=True)
    return int(values[np.argmax(counts)])


def extract_window_features(df, window_size, step_size):
    """Features statistiques par fenetre (mean/std par colonne de signal) --
    memes features que la baseline utilisee pour QRC (confirmee informative,
    F_ratio=0.0997 sur l'ensemble complet du dataset). Retourne (X, y, meta)."""
    x_all, y_all, meta = [], [], []
    df_sorted = df.sort_values([COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP]) if COL_TIMESTAMP in df.columns else df.sort_values([COL_PARTICIPANT, COL_SESSION])
    for (participant, session), group in df_sorted.groupby([COL_PARTICIPANT, COL_SESSION], sort=False):
        x_raw = group[SIGNAL_COLS].values.astype(np.float32)
        y_raw = group[COL_LABEL].values.astype(np.int32)
        for start in range(0, len(x_raw) - window_size + 1, step_size):
            end = start + window_size
            window = x_raw[start:end]
            feat = np.concatenate([window.mean(axis=0), window.std(axis=0)])
            x_all.append(feat)
            y_all.append(_window_label(y_raw[start:end]))
            meta.append({"participant": participant, "session": session, "start": start, "end": end})
    if not x_all:
        return np.empty((0, 2 * len(SIGNAL_COLS))), np.array([], dtype=np.int32), []
    return np.asarray(x_all, dtype=np.float32), np.asarray(y_all, dtype=np.int32), meta


# ============================================================================
# 4. MODELE - SVM A NOYAU PRECALCULE
# ============================================================================
def build_model(params):
    return SVC(kernel="precomputed", C=float(params.get("svm_C", 1.0)),
               class_weight="balanced", probability=False, random_state=42)


def fit_model(model, K_train, y_train):
    model.fit(K_train, y_train)
    return model


def predict_model(model, K_test):
    return model.predict(K_test)


# ============================================================================
# 5. OPTUNA
# ============================================================================
def _suggest_qkernel_params(trial):
    space = QKERNEL_OPTUNA_SPACE
    return {
        "n_qubits": trial.suggest_categorical("n_qubits", space["n_qubits"]),
        "input_scaling": trial.suggest_float("input_scaling", **space["input_scaling"]),
        "svm_C": trial.suggest_float("svm_C", **space["svm_C"]),
        "random_state": 42,
    }


def optuna_objective(trial, df, sessions, window_size, step_size):
    params = _suggest_qkernel_params(trial)
    print(f"[Trial {trial.number}] params: {params}", flush=True)
    scores = []

    for val_part, val_sess in sessions:
        model = None
        try:
            df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
            df_val = df[(df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess)].copy()

            # SMOTE seulement sur df_train, avant le scaling et l'extraction de features
            df_train = resample_dataframe(df_train, SIGNAL_COLS)

            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_val[SIGNAL_COLS] = scaler.transform(df_val[SIGNAL_COLS])

            x_train, y_train, _ = extract_window_features(df_train, window_size, step_size)
            x_val, y_val, _ = extract_window_features(df_val, window_size, step_size)
            if len(y_train) == 0 or len(y_val) == 0:
                scores.append(0.0)
                continue

            encoder = _build_encoder(params, x_train.shape[1])
            states_train = encode_batch(x_train, encoder)
            states_val = encode_batch(x_val, encoder)

            K_train = compute_kernel_matrix(states_train, states_train)
            K_val = compute_kernel_matrix(states_val, states_train)

            model = build_model(params)
            fit_model(model, K_train, y_train)
            y_pred = predict_model(model, K_val)
            scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        except Exception as exc:
            print(f"  [WARN] trial split echoue ({type(exc).__name__}: {exc})")
            scores.append(0.0)
        finally:
            _free_memory(model)

    mean_score = float(np.mean(scores)) if scores else 0.0
    print(f"[Trial {trial.number}] F1-Macro moyen: {mean_score:.4f} | params: {params}", flush=True)
    return mean_score


def optimize_hyperparams(df, n_trials=QKERNEL_OPTUNA_TRIALS):
    print(f"\nOPTUNA QKERNEL - {n_trials} trials | {QKERNEL_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
    random.seed(42)
    config = WINDOW_CONFIGS["default"]
    window_size = config["window_size"]
    step_size = config["step_size"]

    all_sessions = [tuple(x) for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
    val_sessions = random.sample(all_sessions, min(QKERNEL_OPTUNA_SESSIONS, len(all_sessions)))
    print(f"Sessions Optuna: {val_sessions}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,
            n_startup_trials=10,
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=3,
            reduction_factor=3,
        ),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, df, val_sessions, window_size, step_size),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=10,
        gc_after_trial=True,
        catch=(Exception,),
    )

    print(f"\nBest F1-Macro: {study.best_value:.4f}")
    print(f"Best params  : {study.best_params}")

    OPTUNA_PATH_QKERNEL.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_QKERNEL, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return {**MODEL_PARAMS["QKERNEL"], **study.best_params}


# ============================================================================
# 6. VISUALISATION
# ============================================================================
def plot_fold(fold_idx, test_part, test_sess, save_dir, f1_mac, bal_acc, y_true, y_pred):
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"QKERNEL | Fold {fold_idx} - P{test_part} S{test_sess}"
        f" | F1-Macro: {f1_mac:.3f} | Bal.Acc: {bal_acc:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    axes[0].bar(["F1-Macro", "Balanced Acc"], [f1_mac, bal_acc], color=["#2563EB", "#059669"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Scores du fold")
    axes[0].grid(True, axis="y", alpha=0.3)

    cm = confusion_matrix(y_true, y_pred)
    labels = [TARGET_NAMES[i] for i in np.unique(np.concatenate([y_true, y_pred]))]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
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

    plt.tight_layout()
    plot_path = save_dir / f"fold{fold_idx:02d}_P{test_part}_S{test_sess}_curves.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Courbes sauvegardees: {plot_path}")


# ============================================================================
# 7. MODELE GLOBAL
# ============================================================================
def _compute_classification_metrics(y_true, y_pred):
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
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


def train_global_model(df_labeled, df_unlabeled, best_params):
    print("\n" + "=" * 60 + f"\nMODELE GLOBAL - {MODEL_NAME}\n" + "=" * 60)
    config = WINDOW_CONFIGS["default"]
    window_size, step_size = config["window_size"], config["step_size"]

    try:
        unique_sessions_all = [
            tuple(x) for x in df_labeled[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
        ]
        idx = np.random.default_rng(42).permutation(len(unique_sessions_all))
        split_point = int(0.9 * len(unique_sessions_all))
        train_sessions = [unique_sessions_all[i] for i in idx[:split_point]]
        val_sessions = [unique_sessions_all[i] for i in idx[split_point:]]

        df_tr_raw = df_labeled[
            df_labeled.set_index([COL_PARTICIPANT, COL_SESSION]).index.isin(train_sessions)
        ].copy()
        df_vl_raw = df_labeled[
            df_labeled.set_index([COL_PARTICIPANT, COL_SESSION]).index.isin(val_sessions)
        ].copy()

        # SMOTE seulement sur le train, avant scaling et extraction de features
        df_tr_raw = resample_dataframe(df_tr_raw, SIGNAL_COLS)

        scaler = RobustScaler()
        df_tr_raw[SIGNAL_COLS] = scaler.fit_transform(df_tr_raw[SIGNAL_COLS])
        df_vl_raw[SIGNAL_COLS] = scaler.transform(df_vl_raw[SIGNAL_COLS])

        x_train, y_train, _ = extract_window_features(df_tr_raw, window_size, step_size)
        x_val, y_val, _ = extract_window_features(df_vl_raw, window_size, step_size)

        if len(y_train) == 0 or len(y_val) == 0:
            print("  Modele global echoue : train/val vide")
            return

        encoder = _build_encoder(best_params, x_train.shape[1])
        states_train = encode_batch(x_train, encoder)
        states_val = encode_batch(x_val, encoder)

        K_train = compute_kernel_matrix(states_train, states_train)
        K_val = compute_kernel_matrix(states_val, states_train)

        model = build_model(best_params)
        fit_model(model, K_train, y_train)
        y_pred = predict_model(model, K_val)

        f1_mac = f1_score(y_val, y_pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        print(f"  F1-Macro (val globale) : {f1_mac:.4f} | Bal.Acc : {bal_acc:.4f}")

        models_dir = MODELS_DIR / "QKERNEL"
        models_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = models_dir / f"{MODEL_NAME}_global.pkl"
        # support_states : etats quantiques encodes du train, necessaires pour
        # calculer le noyau test x train a l'inference sur de nouvelles fenetres.
        joblib.dump({"encoder": encoder, "svm": model, "scaler": scaler,
                     "support_states": states_train}, pkl_path)
        print(f"  Pickle backup : {pkl_path}")

        edge_metrics = {
            "native_qkernel": _compute_classification_metrics(y_val, y_pred)
        }
        edge_metrics["native_qkernel"]["model_size_kb"] = round(pkl_path.stat().st_size / 1024, 2)
        edge_metrics["native_qkernel"]["deployment_status"] = (
            "export_stm32_non_tente : necessiterait d'embarquer les etats quantiques "
            "de TOUS les vecteurs de support (potentiellement des centaines), "
            "structure differente des modeles a poids fixes deja exportes (QRC). "
            "Faisable en principe (encodage + produits scalaires + decision lineaire "
            "SVM, tous compatibles avec les operateurs stedgeai deja confirmes), "
            "mais non tente ici -- a explorer separement si les resultats F1 le justifient."
        )

        _save_edge_comparison_metrics(MODEL_NAME, edge_metrics)

    except Exception as exc:
        print(f"  Modele global echoue ({type(exc).__name__}: {exc})")
        traceback.print_exc()


# ============================================================================
# 8. BOUCLE LOSO
# ============================================================================
def main():
    print("\n" + "=" * 60 + f"\nTRAIN LOSO - {MODEL_NAME}\n" + "=" * 60)
    if not DATA_MODEL_READY.exists():
        return print(f"Dataset non trouve: {DATA_MODEL_READY}")

    df = pd.read_csv(DATA_MODEL_READY)
    df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors="coerce").fillna(-1).astype(int)
    n_unlabeled = int((df[COL_LABEL] == -1).sum())
    df_labeled = df[df[COL_LABEL] >= 0].copy()
    df_unlabeled = df[df[COL_LABEL] == -1].copy()
    print(f"Labeled: {len(df_labeled)} lignes | Unlabeled exclues: {n_unlabeled} lignes")

    unique_sessions = [
        tuple(x)
        for x in df_labeled[df_labeled[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]]
        .drop_duplicates()
        .values
    ]

    best_params = optimize_hyperparams(df_labeled) if USE_OPTUNA_QKERNEL else MODEL_PARAMS["QKERNEL"]
    best_params = {**MODEL_PARAMS["QKERNEL"], **best_params}

    config = WINDOW_CONFIGS["default"]
    window_size = config["window_size"]
    step_size = config["step_size"]

    all_metrics = []
    plots_dir = BASE_DIR / "training_curves" / MODEL_NAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (test_part, test_sess) in enumerate(unique_sessions, start=1):
        print(f"\n--- FOLD {fold_idx}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")

        df_train = df_labeled[~((df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess))].copy()
        df_test = df_labeled[(df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess)].copy()

        model = None
        try:
            # SMOTE seulement sur df_train, avant le scaling et l'extraction de features
            df_train = resample_dataframe(df_train, SIGNAL_COLS)

            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])

            x_train, y_train, _ = extract_window_features(df_train, window_size, step_size)
            x_test, y_test, _ = extract_window_features(df_test, window_size, step_size)

            if len(y_train) == 0 or len(y_test) == 0:
                print("  WARNING: train/test vide. Fold ignore.")
                continue

            print(f"  Train: {len(x_train)} fenetres | Test: {len(x_test)} fenetres")

            encoder = _build_encoder(best_params, x_train.shape[1])
            states_train = encode_batch(x_train, encoder)
            states_test = encode_batch(x_test, encoder)

            K_train = compute_kernel_matrix(states_train, states_train)
            # test x train (PAS test x test) : un SVM a noyau precalcule a
            # besoin du noyau entre les points de test et les vecteurs de
            # support/train, pas entre les points de test eux-memes.
            K_test = compute_kernel_matrix(states_test, states_train)

            model = build_model(best_params)
            fit_model(model, K_train, y_train)
            y_pred = predict_model(model, K_test)

            f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            labels_present = np.unique(np.concatenate([y_test, y_pred]))
            report = cast(str, classification_report(
                y_test,
                y_pred,
                labels=labels_present,
                target_names=[TARGET_NAMES[i] for i in labels_present],
                zero_division=0,
                output_dict=False,
            ))
            print("\n" + report)

            plot_fold(fold_idx, test_part, test_sess, plots_dir, f1_mac, bal_acc, y_test, y_pred)

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

    df_res = pd.DataFrame(all_metrics)
    print("\n" + "=" * 60 + f"\nRESULTATS FINAUX - {MODEL_NAME}\n" + "=" * 60)
    print(df_res.describe().loc[["mean", "std"]])

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

    train_global_model(df_labeled, df_unlabeled, best_params)


if __name__ == "__main__":
    main()
