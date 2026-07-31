import gc
import json
import random
import sys
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
# pyrefly: ignore [missing-import]
from reservoirpy.nodes import Ridge


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import *

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 1. CONSTANTES
# ============================================================================
MODEL_NAME = "QRC"
LABEL_MAPPING = {-1: 0, 0: 1, 1: 2}
TARGET_NAMES = ["baseline (-1)", "activity (0)", "fatigue (1)"]

# QRC est plus coûteux que ESN: chaque fenêtre simule un circuit quantique.
# On garde le pipeline LOSO, mais avec des fenêtres adaptées au coût du QRC.
QRC_WINDOW_SIZE = 192
QRC_STEP_SIZE = 96
QRC_PROGRESS_EVERY_SESSIONS = 3


# ============================================================================
# 2. QRC CORE - SIMULATEUR STATE-VECTOR NUMPY
# ============================================================================
def _free_memory(*objects):
    for obj in objects:
        del obj
    gc.collect()


def _rx(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex64)


def _ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex64)


def _rz(theta):
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex64,
    )


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


def _measure_z_all(state, n_qubits):
    probs = np.abs(state) ** 2
    values = np.empty(n_qubits, dtype=np.float32)
    dim = len(state)
    for q in range(n_qubits):
        mask = 1 << (n_qubits - 1 - q)
        signs = np.fromiter(
            (1.0 if (idx & mask) == 0 else -1.0 for idx in range(dim)),
            dtype=np.float32,
            count=dim,
        )
        values[q] = float(np.dot(probs, signs))
    return values


def _build_qrc(params, input_dim):
    """
    Initialise un reservoir quantique fixe.

    Les poids quantiques ne sont pas entraînés: le readout Ridge apprend seulement
    sur les mesures produites par le reservoir, comme en Reservoir Computing.
    """
    n_qubits = int(params["n_qubits"])
    n_layers = int(params["n_layers"])
    rng = np.random.default_rng(int(params.get("random_state", 42)))

    cnot_pairs = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    cnot_perms = [_cnot_permutation(n_qubits, c, t) for c, t in cnot_pairs]
    z_signs = []
    dim = 2 ** n_qubits
    for q in range(n_qubits):
        mask = 1 << (n_qubits - 1 - q)
        z_signs.append(
            np.asarray([1.0 if (idx & mask) == 0 else -1.0 for idx in range(dim)], dtype=np.float32)
        )

    return {
        "params": dict(params),
        "input_dim": int(input_dim),
        "input_weights": rng.normal(0.0, 1.0, size=(input_dim, n_qubits)).astype(np.float32),
        "input_bias": rng.uniform(-np.pi, np.pi, size=n_qubits).astype(np.float32),
        "feedback_weights": rng.normal(0.0, 1.0, size=(n_qubits, n_qubits)).astype(np.float32),
        "reservoir_angles": rng.uniform(
            -np.pi,
            np.pi,
            size=(n_layers, n_qubits, 3),
        ).astype(np.float32),
        "cnot_perms": cnot_perms,
        "z_signs": z_signs,
    }


def _initial_state(n_qubits):
    state = np.zeros(2 ** n_qubits, dtype=np.complex64)
    state[0] = 1.0 + 0.0j
    return state


def _qrc_step(x_t, state, prev_features, reservoir):
    params = reservoir["params"]
    n_qubits = int(params["n_qubits"])
    input_scaling = float(params["input_scaling"])
    reservoir_scale = float(params["reservoir_scale"])
    feedback_scale = float(params["feedback_scale"])
    leak_rate = float(params["leak_rate"])

    angles = (
        input_scaling * (x_t @ reservoir["input_weights"])
        + feedback_scale * (prev_features @ reservoir["feedback_weights"])
        + reservoir["input_bias"]
    )

    for q, angle in enumerate(angles):
        state = _apply_single_qubit_gate(state, _ry(float(angle)), q, n_qubits)

    for layer_angles in reservoir["reservoir_angles"]:
        for q, (rx, ry, rz) in enumerate(layer_angles):
            state = _apply_single_qubit_gate(state, _rx(float(rx) * reservoir_scale), q, n_qubits)
            state = _apply_single_qubit_gate(state, _ry(float(ry) * reservoir_scale), q, n_qubits)
            state = _apply_single_qubit_gate(state, _rz(float(rz) * reservoir_scale), q, n_qubits)
        for perm in reservoir["cnot_perms"]:
            state = _apply_permutation(state, perm)

    norm = np.linalg.norm(state)
    if norm > 0:
        state = state / norm

    probs = np.abs(state) ** 2
    measured = np.asarray([float(np.dot(probs, signs)) for signs in reservoir["z_signs"]], dtype=np.float32)
    features = ((1.0 - leak_rate) * prev_features + leak_rate * measured).astype(np.float32)
    return state, features


def _qrc_states(window, reservoir):
    params = reservoir["params"]
    n_qubits = int(params["n_qubits"])
    temporal_stride = max(1, int(params.get("temporal_stride", 1)))
    state = _initial_state(n_qubits)
    prev_features = np.zeros(n_qubits, dtype=np.float32)
    states = []

    for x_t in window[::temporal_stride]:
        state, prev_features = _qrc_step(x_t.astype(np.float32), state, prev_features, reservoir)
        states.append(prev_features.copy())

    if not states:
        states.append(prev_features.copy())
    return np.asarray(states, dtype=np.float32)


def _summarize_states(states, params):
    summary = params.get("state_summary", "last_mean_std")
    last = states[-1]

    if summary == "last":
        return last.astype(np.float32)
    if summary == "last_mean":
        return np.concatenate([last, states.mean(axis=0)]).astype(np.float32)
    if summary == "last_mean_std":
        return np.concatenate([last, states.mean(axis=0), states.std(axis=0)]).astype(np.float32)

    raise ValueError(f"state_summary inconnu: {summary}")


def _transform_window(window, reservoir):
    states = _qrc_states(window, reservoir)
    return _summarize_states(states, reservoir["params"])


# ============================================================================
# 3. DONNEES
# ============================================================================
def _sort_df(df):
    if COL_TIMESTAMP in df.columns:
        return df.sort_values([COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP])
    return df.sort_values([COL_PARTICIPANT, COL_SESSION])


def _window_label(y_window):
    values, counts = np.unique(y_window, return_counts=True)
    return int(values[np.argmax(counts)])


def _qrc_window_config():
    return {"window_size": QRC_WINDOW_SIZE, "step_size": QRC_STEP_SIZE}


def _count_group_windows(groups, window_size, step_size):
    total = 0
    for _, group in groups:
        n_rows = len(group)
        if n_rows >= window_size:
            total += ((n_rows - window_size) // step_size) + 1
    return total


def extract_qrc_windows(df, reservoir, window_size, step_size, tag=None, verbose=False):
    x_all, y_all = [], []
    df = _sort_df(df)
    groups = list(df.groupby([COL_PARTICIPANT, COL_SESSION], sort=False))
    total_windows = _count_group_windows(groups, window_size, step_size)

    if verbose:
        label = f" {tag}" if tag else ""
        print(
            f"  Extraction QRC{label}: {len(groups)} sessions, ~{total_windows} fenetres "
            f"(window={window_size}, step={step_size}, stride={reservoir['params'].get('temporal_stride')})",
            flush=True,
        )

    done_windows = 0
    for idx, ((participant, session), group) in enumerate(groups, start=1):
        x_raw = group[SIGNAL_COLS].values.astype(np.float32)
        y_raw = group[COL_LABEL].values.astype(np.int32)
        for start in range(0, len(x_raw) - window_size + 1, step_size):
            end = start + window_size
            x_all.append(_transform_window(x_raw[start:end], reservoir))
            y_all.append(_window_label(y_raw[start:end]))
            done_windows += 1

        if verbose and (
            idx == 1
            or idx == len(groups)
            or idx % QRC_PROGRESS_EVERY_SESSIONS == 0
        ):
            print(
                f"    {tag or 'QRC'}: session {idx}/{len(groups)} "
                f"(P{participant} S{session}) | {done_windows}/{total_windows} fenetres",
                flush=True,
            )

    if not x_all:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int32)
    return np.asarray(x_all, dtype=np.float32), np.asarray(y_all, dtype=np.int32)


def first_window_states(df, reservoir, window_size):
    df = _sort_df(df)
    for _, group in df.groupby([COL_PARTICIPANT, COL_SESSION], sort=False):
        x_raw = group[SIGNAL_COLS].values.astype(np.float32)
        if len(x_raw) >= window_size:
            return _qrc_states(x_raw[:window_size], reservoir)
    return None


# ============================================================================
# 4. MODELE, OPTUNA
# ============================================================================
def build_model(params):
    return Ridge(ridge=float(params["ridge_alpha"]), name="readout")


def _sample_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = dict(zip(classes, weights))
    return np.asarray([weight_map[y] for y in y_train], dtype=np.float32)


def _to_onehot(y, num_classes=None):
    num_classes = num_classes or len(LABEL_MAPPING)
    y_onehot = np.zeros((len(y), num_classes), dtype=np.float32)
    y_onehot[np.arange(len(y)), y.astype(int)] = 1.0
    return y_onehot


def fit_readout(model, x_train, y_train):
    weights = np.sqrt(_sample_weights(y_train)).reshape(-1, 1)
    x_weighted = x_train * weights
    y_weighted = _to_onehot(y_train) * weights
    model.fit(x_weighted, y_weighted)
    return model


def predict_readout(model, x):
    scores = np.asarray(model.run(x), dtype=np.float32)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    return np.argmax(scores, axis=1).astype(np.int32)


def _suggest_qrc_params(trial):
    space = QRC_OPTUNA_SPACE
    return {
        "n_qubits": trial.suggest_categorical("n_qubits", space["n_qubits"]),
        "n_layers": trial.suggest_categorical("n_layers", space["n_layers"]),
        "input_scaling": trial.suggest_float("input_scaling", **space["input_scaling"]),
        "reservoir_scale": trial.suggest_float("reservoir_scale", **space["reservoir_scale"]),
        "feedback_scale": trial.suggest_float("feedback_scale", **space["feedback_scale"]),
        "leak_rate": trial.suggest_float("leak_rate", **space["leak_rate"]),
        "ridge_alpha": trial.suggest_float("ridge_alpha", **space["ridge_alpha"]),
        "temporal_stride": trial.suggest_categorical("temporal_stride", space["temporal_stride"]),
        "state_summary": trial.suggest_categorical("state_summary", space["state_summary"]),
        "random_state": 42,
    }


def optuna_objective(trial, df, sessions, window_size, step_size):
    params = _suggest_qrc_params(trial)
    scores = []

    for val_part, val_sess in sessions:
        model = None
        try:
            df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
            df_val = df[(df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess)].copy()

            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_val[SIGNAL_COLS] = scaler.transform(df_val[SIGNAL_COLS])

            reservoir = _build_qrc(params, len(SIGNAL_COLS))
            x_train, y_train = extract_qrc_windows(df_train, reservoir, window_size, step_size)
            x_val, y_val = extract_qrc_windows(df_val, reservoir, window_size, step_size)
            if len(y_train) == 0 or len(y_val) == 0:
                scores.append(0.0)
                continue

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


def optimize_hyperparams(df, n_trials=QRC_OPTUNA_TRIALS):
    print(f"\nOPTUNA QRC - {n_trials} trials | {QRC_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
    random.seed(42)
    config = _qrc_window_config()
    window_size = config["window_size"]
    step_size = config["step_size"]

    all_sessions = [tuple(x) for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
    val_sessions = random.sample(all_sessions, min(QRC_OPTUNA_SESSIONS, len(all_sessions)))
    print(f"Sessions Optuna: {val_sessions}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,
            n_startup_trials=20,
        ),
        pruner=optuna.pruners.HyperbandPruner(  # ← CORRIGÉ : plus efficace
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

    OPTUNA_PATH_QRC.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_QRC, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return {**MODEL_PARAMS["QRC"], **study.best_params}


# ============================================================================
# 5. VISUALISATION
# ============================================================================
def plot_fold(fold_idx, test_part, test_sess, save_dir, f1_mac, bal_acc, y_true, y_pred, states):
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        f"QRC | Fold {fold_idx} - P{test_part} S{test_sess}"
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

    if states is not None and len(states) > 0:
        axes[2].plot(states, linewidth=1.7)
        axes[2].set_title("Mesures Pauli-Z du reservoir")
        axes[2].set_xlabel("Temps apres stride")
        axes[2].set_ylabel("<Z>")
        axes[2].set_ylim(-1.05, 1.05)
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
def train_global_model(df_labeled, df_unlabeled, best_params):
    print("\n" + "=" * 60 + f"\nMODELE GLOBAL - {MODEL_NAME}\n" + "=" * 60)
    print(
        f"  ⚠ {MODEL_NAME} : pas d'export STM32 (.tflite/.h). "
        f"Métriques LOSO conservées ({len(df_labeled)} labellisées, {len(df_unlabeled)} unlabeled)."
    )


# ============================================================================
# 7. BOUCLE LOSO
# ============================================================================
def main():
    print("\n" + "=" * 60 + f"\nTRAIN LOSO - {MODEL_NAME}\n" + "=" * 60)
    if not DATA_MODEL_READY.exists():
        return print(f"Dataset non trouve: {DATA_MODEL_READY}")

    df = pd.read_csv(DATA_MODEL_READY)
    df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors="coerce").fillna(-1).astype(int)
    df_labeled = df[df[COL_LABEL] >= 0].copy()
    df_unlabeled = df[df[COL_LABEL] == -1].copy()
    print(f"Labeled: {len(df_labeled)} lignes | Unlabeled: {len(df_unlabeled)} lignes")

    unique_sessions = [
        tuple(x)
        for x in df_labeled[df_labeled[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]]
        .drop_duplicates()
        .values
    ]

    best_params = optimize_hyperparams(df_labeled) if USE_OPTUNA_QRC else MODEL_PARAMS["QRC"]
    best_params = {**MODEL_PARAMS["QRC"], **best_params}

    all_metrics = []
    plots_dir = BASE_DIR / "training_curves" / MODEL_NAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (test_part, test_sess) in enumerate(unique_sessions, start=1):
        print(f"\n--- FOLD {fold_idx}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")
        config = _qrc_window_config()
        window_size = config["window_size"]
        step_size = config["step_size"]

        df_train = df_labeled[~((df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess))].copy()
        df_test = df_labeled[(df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess)].copy()

        model = None
        try:
            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])

            reservoir = _build_qrc(best_params, len(SIGNAL_COLS))
            x_train, y_train = extract_qrc_windows(
                df_train,
                reservoir,
                window_size,
                step_size,
                tag="train",
                verbose=True,
            )
            x_test, y_test = extract_qrc_windows(
                df_test,
                reservoir,
                window_size,
                step_size,
                tag="test",
                verbose=True,
            )

            if len(y_train) == 0 or len(y_test) == 0:
                print("  WARNING: train/test vide. Fold ignore.")
                continue

            print(f"  Train: {len(x_train)} fenetres | Test: {len(x_test)} fenetres")

            model = build_model(best_params)
            fit_readout(model, x_train, y_train)
            y_pred = predict_readout(model, x_test)

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
                ),)
            print("\n" + report)
            
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
