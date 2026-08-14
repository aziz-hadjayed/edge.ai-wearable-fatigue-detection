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
# pyrefly: ignore [missing-import]
import onnx
# pyrefly: ignore [missing-import]
import onnx.helper
# pyrefly: ignore [missing-import]
import onnx.numpy_helper
# pyrefly: ignore [missing-import]
import onnxruntime as ort
import optuna
import pandas as pd
# pyrefly: ignore [missing-import]
from reservoirpy.nodes import Ridge


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
from sklearn.utils.class_weight import compute_class_weight

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 1. CONSTANTES
# ============================================================================
MODEL_NAME = "QRC"
LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES = ["baseline", "activity", "pre_fatigue", "fatigue"]

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


def _single_qubit_gate_matrix(gate, qubit, n_qubits):
    """Construit la matrice 2^n x 2^n appliquant `gate` (2x2) au qubit `qubit`
    parmi n_qubits, identite ailleurs — via produit de Kronecker."""
    mats = [np.eye(2, dtype=np.complex64)] * n_qubits
    mats[qubit] = gate
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result.astype(np.complex64)


def _permutation_matrix(perm):
    """Convertit un tableau de permutation d'indices en matrice de
    permutation 2^n x 2^n."""
    dim = len(perm)
    mat = np.zeros((dim, dim), dtype=np.complex64)
    mat[perm, np.arange(dim)] = 1.0
    return mat


def _build_fixed_layer_matrix(reservoir_angles, cnot_perms, n_qubits, reservoir_scale):
    dim = 2 ** n_qubits
    U = np.eye(dim, dtype=np.complex64)

    for layer_angles in reservoir_angles:
        for q, (rx, ry, rz) in enumerate(layer_angles):
            rx_b = np.pi * np.tanh(float(rx) * reservoir_scale / np.pi)
            ry_b = np.pi * np.tanh(float(ry) * reservoir_scale / np.pi)
            rz_b = np.pi * np.tanh(float(rz) * reservoir_scale / np.pi)
            U = _single_qubit_gate_matrix(_rx(rx_b), q, n_qubits) @ U
            U = _single_qubit_gate_matrix(_ry(ry_b), q, n_qubits) @ U
            U = _single_qubit_gate_matrix(_rz(rz_b), q, n_qubits) @ U
        for perm in cnot_perms:
            U = _permutation_matrix(perm) @ U

    return U.astype(np.complex64)


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

    # Ordre des tirages RNG preserve a l'identique de la version precedente
    # (input_weights, input_bias, feedback_weights, PUIS reservoir_angles) :
    # affecter reservoir_angles a une variable locale ici (au lieu de l'inliner
    # dans le dict retourne comme avant) ne change aucun tirage aleatoire pour
    # un random_state donne, seule la matrice fixe precalculee est nouvelle.
    input_weights = rng.normal(0.0, 1.0, size=(input_dim, n_qubits)).astype(np.float32)
    input_bias = rng.uniform(-np.pi, np.pi, size=n_qubits).astype(np.float32)
    feedback_weights = rng.normal(0.0, 1.0, size=(n_qubits, n_qubits)).astype(np.float32)
    reservoir_angles = rng.uniform(
        -np.pi,
        np.pi,
        size=(n_layers, n_qubits, 3),
    ).astype(np.float32)
    fixed_layer_matrix = _build_fixed_layer_matrix(
        reservoir_angles=reservoir_angles,
        cnot_perms=cnot_perms,
        n_qubits=n_qubits,
        reservoir_scale=float(params["reservoir_scale"]),
    )

    return {
        "params": dict(params),
        "input_dim": int(input_dim),
        "input_weights": input_weights,
        "input_bias": input_bias,
        "feedback_weights": feedback_weights,
        "reservoir_angles": reservoir_angles,
        "cnot_perms": cnot_perms,
        "fixed_layer_matrix": fixed_layer_matrix,
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
    feedback_scale = float(params["feedback_scale"])
    leak_rate = float(params["leak_rate"])

    angles = (
        input_scaling * (x_t @ reservoir["input_weights"])
        + feedback_scale * (prev_features @ reservoir["feedback_weights"])
        + reservoir["input_bias"]
    )
    # Bornage smooth (tanh, pas un clip dur) pour eviter l'aliasing de
    # periode 2*pi de la rotation RY : garantit angle dans (-pi, pi),
    # strictement monotone donc preserve l'ordre relatif du signal meme
    # pour les valeurs extremes, contrairement a un clip qui ecraserait
    # toutes les valeurs au-dela du seuil en une valeur identique.
    angles = np.pi * np.tanh(angles / np.pi)

    for q, angle in enumerate(angles):
        state = _apply_single_qubit_gate(state, _ry(float(angle)), q, n_qubits)

    state = reservoir["fixed_layer_matrix"] @ state

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
    print(f"[Trial {trial.number}] params: {params}", flush=True)
    scores = []

    for val_part, val_sess in sessions:
        model = None
        try:
            df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
            df_val = df[(df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess)].copy()

            # SMOTE seulement sur df_train, avant le scaling et l'extraction de fenetres
            df_train = resample_dataframe(df_train, SIGNAL_COLS)

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

    mean_score = float(np.mean(scores)) if scores else 0.0
    print(f"[Trial {trial.number}] F1-Macro moyen: {mean_score:.4f} | params: {params}", flush=True)
    return mean_score


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
# 5b. EXPORT ONNX QRC
# ============================================================================
def _moveaxis_perm_onnx(qubit, n_qubits):
    """perm tel que Transpose(tensor, perm) == np.moveaxis(tensor, qubit, 0)."""
    return [qubit] + [i for i in range(n_qubits) if i != qubit]


def _inverse_moveaxis_perm_onnx(qubit, n_qubits):
    """perm tel que Transpose(tensor, perm) == np.moveaxis(tensor, 0, qubit)
    en partant d'un tensor deja permute par _moveaxis_perm_onnx(qubit, n_qubits)."""
    forward = _moveaxis_perm_onnx(qubit, n_qubits)
    inverse = [0] * n_qubits
    for new_pos, old_axis in enumerate(forward):
        inverse[old_axis] = new_pos
    return inverse


def _apply_qubit_gate_onnx_nodes(nodes, in_name, gate_name, qubit, n_qubits, shape_names, out_name):
    """
    Emet les noeuds ONNX appliquant un gate reel 2x2 (`gate_name`, valeur
    dynamique calculee dans le graphe) au qubit `qubit` parmi n_qubits d'un
    etat reel 1D (`in_name`, forme (2**n_qubits,)) -- traduction directe de
    _apply_single_qubit_gate (reshape/moveaxis/tensordot -> Reshape/Transpose/
    MatMul), validee numeriquement en isolation dans
    /tmp/prototype_qrc_onnx_gate.py (diff max 2.4e-07 sur n_qubits 3-8, toutes
    positions de qubit) avant integration ici.

    n_qubits et qubit sont STATIQUES (connus a la construction du graphe, pas
    a l'execution) : perms et shapes sont donc des constantes Python, pas de
    boucle/Kronecker runtime necessaire.
    """
    perm_fwd = _moveaxis_perm_onnx(qubit, n_qubits)
    perm_inv = _inverse_moveaxis_perm_onnx(qubit, n_qubits)
    t1, t2, t3, t4, t5 = [f"{out_name}__t{i}" for i in range(1, 6)]

    nodes.append(onnx.helper.make_node("Reshape", [in_name, shape_names["nqubits"]], [t1]))
    nodes.append(onnx.helper.make_node("Transpose", [t1], [t2], perm=perm_fwd))
    nodes.append(onnx.helper.make_node("Reshape", [t2, shape_names["flat"]], [t3]))
    nodes.append(onnx.helper.make_node("MatMul", [gate_name, t3], [t4]))
    nodes.append(onnx.helper.make_node("Reshape", [t4, shape_names["nqubits"]], [t5]))
    nodes.append(onnx.helper.make_node("Transpose", [t5], [out_name + "__t6"], perm=perm_inv))
    nodes.append(onnx.helper.make_node("Reshape", [out_name + "__t6", shape_names["flat_out"]], [out_name]))


def _build_onnx_qrc_graph(reservoir, readout_model, model_name):
    """
    Construit un graphe ONNX unique reproduisant le forward QRC complet
    (rotation d'encodage d'entree dependante de x_t/prev_features + matrice
    fixe precalculee `fixed_layer_matrix` + mesure Pauli-Z + resume + readout
    Ridge), pour UNE fenetre a la fois (pas de dimension batch : le chemin
    Python _qrc_states/extract_qrc_windows traite deja les fenetres une par
    une, meme convention ici).

    Etat quantique represente par DEUX tenseurs reels (real/imag) -- ONNX n'a
    pas de dtype complexe portable. fixed_layer_matrix (complexe, precalculee
    une fois par _build_qrc) est separee en deux constantes U_real_T/U_imag_T
    (deja TRANSPOSEES a la construction pour eviter un Transpose a
    l'execution, cf. convention MatMul(vecteur_1D, matrice) = tensordot).

    Boucle sur les pas de temps (window[::temporal_stride]) DEROULEE en
    Python a la construction du graphe (n_steps est statique), comme le fait
    deja _build_onnx_mesn_graph (MESN.py) pour sa recurrence.
    """
    params = reservoir["params"]
    n_qubits = int(params["n_qubits"])
    input_dim = int(reservoir["input_dim"])
    temporal_stride = max(1, int(params.get("temporal_stride", 1)))
    input_scaling = float(params["input_scaling"])
    feedback_scale = float(params["feedback_scale"])
    leak_rate = float(params["leak_rate"])
    state_summary = params.get("state_summary", "last_mean_std")
    window_size = QRC_WINDOW_SIZE
    dim = 2 ** n_qubits

    step_indices = list(range(0, window_size, temporal_stride))
    n_steps = len(step_indices)
    if n_steps == 0:
        raise ValueError("temporal_stride >= window_size : aucun pas de temps, export ONNX impossible.")

    try:
        Wout = np.asarray(readout_model.Wout, dtype=np.float32)
        bias = np.asarray(readout_model.bias, dtype=np.float32).reshape(-1)
    except AttributeError:
        print(
            f"  [DEBUG] extraction des poids du readout QRC echouee. "
            f"Attributs disponibles: {dir(readout_model)}"
        )
        raise

    U_fixed = np.asarray(reservoir["fixed_layer_matrix"], dtype=np.complex64)
    # Deja transposees : MatMul(state_1D, U_T) == tensordot(U, state, axes=([1],[0]))
    # == U @ state, convention vecteur-ligne (meme raisonnement que W_in_T/W_T
    # dans _build_onnx_mesn_graph).
    U_real_T = np.ascontiguousarray(U_fixed.real.T.astype(np.float32))
    U_imag_T = np.ascontiguousarray(U_fixed.imag.T.astype(np.float32))

    nodes = []
    initializers = [
        onnx.numpy_helper.from_array(np.asarray(reservoir["input_weights"], dtype=np.float32), "input_weights"),
        onnx.numpy_helper.from_array(np.asarray(reservoir["feedback_weights"], dtype=np.float32), "feedback_weights"),
        onnx.numpy_helper.from_array(np.asarray(reservoir["input_bias"], dtype=np.float32), "input_bias"),
        onnx.numpy_helper.from_array(U_real_T, "U_real_T"),
        onnx.numpy_helper.from_array(U_imag_T, "U_imag_T"),
        onnx.numpy_helper.from_array(np.asarray([input_scaling], dtype=np.float32), "input_scaling"),
        onnx.numpy_helper.from_array(np.asarray([feedback_scale], dtype=np.float32), "feedback_scale"),
        onnx.numpy_helper.from_array(np.asarray([1.0 - leak_rate], dtype=np.float32), "one_minus_leak"),
        onnx.numpy_helper.from_array(np.asarray([leak_rate], dtype=np.float32), "leak_rate"),
        onnx.numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), "half"),
        onnx.numpy_helper.from_array(np.asarray([-1.0], dtype=np.float32), "neg_one"),
        onnx.numpy_helper.from_array(np.asarray([np.pi], dtype=np.float32), "pi"),
        onnx.numpy_helper.from_array(np.asarray(Wout, dtype=np.float32), "Wout"),
        onnx.numpy_helper.from_array(bias, "bias"),
        onnx.numpy_helper.from_array(_initial_state(n_qubits).real.astype(np.float32), "state_real_init"),
        onnx.numpy_helper.from_array(np.zeros(dim, dtype=np.float32), "state_imag_init"),
        onnx.numpy_helper.from_array(np.zeros(n_qubits, dtype=np.float32), "prev_features_init"),
    ]

    shape_names = {"nqubits": "shape_nqubits", "flat": "shape_flat", "flat_out": "shape_flat_out"}
    initializers.extend(
        [
            onnx.numpy_helper.from_array(np.asarray([2] * n_qubits, dtype=np.int64), shape_names["nqubits"]),
            onnx.numpy_helper.from_array(np.asarray([2, dim // 2], dtype=np.int64), shape_names["flat"]),
            onnx.numpy_helper.from_array(np.asarray([dim], dtype=np.int64), shape_names["flat_out"]),
        ]
    )
    for q in range(n_qubits):
        initializers.append(onnx.numpy_helper.from_array(np.asarray([q], dtype=np.int64), f"idx_q{q}"))
    for t in range(n_steps):
        initializers.append(
            onnx.numpy_helper.from_array(np.asarray([step_indices[t]], dtype=np.int64), f"idx_t{t}")
        )
    initializers.append(onnx.numpy_helper.from_array(np.asarray([input_dim], dtype=np.int64), "shape_features"))

    # --- normalisation RobustScaler du sous-echantillonnage : PAS appliquee
    # ici -- contrairement a MESN, QRC n'a pas de RobustScaler par capteur
    # integre au graphe dans la version Python actuelle (extract_qrc_windows
    # recoit un df DEJA scale par le RobustScaler *global* du fold, cf. main()/
    # optuna_objective). Le graphe QRC prend donc en entree une fenetre DEJA
    # normalisee (meme convention que le chemin Python), pas de bloc
    # scaler ici -- a la difference de MESN qui integre le scaler par capteur
    # car son graphe prend des fenetres brutes en entree.
    graph_inputs = [
        onnx.helper.make_tensor_value_info("x_window", onnx.TensorProto.FLOAT, [window_size, input_dim])
    ]

    prev_features = "prev_features_init"
    state_real = "state_real_init"
    state_imag = "state_imag_init"
    kept_states = []

    for t in range(n_steps):
        prefix = f"t{t}"
        x_t_raw = f"{prefix}_x_t_raw"
        x_t = f"{prefix}_x_t"
        nodes.append(onnx.helper.make_node("Gather", ["x_window", f"idx_t{t}"], [x_t_raw], axis=0))
        nodes.append(onnx.helper.make_node("Reshape", [x_t_raw, "shape_features"], [x_t]))

        # angles = input_scaling*(x_t @ input_weights) + feedback_scale*(prev_features @ feedback_weights) + input_bias
        in_proj = f"{prefix}_in_proj"
        fb_proj = f"{prefix}_fb_proj"
        in_proj_scaled = f"{prefix}_in_proj_scaled"
        fb_proj_scaled = f"{prefix}_fb_proj_scaled"
        angles_sum1 = f"{prefix}_angles_sum1"
        angles = f"{prefix}_angles"
        nodes.append(onnx.helper.make_node("MatMul", [x_t, "input_weights"], [in_proj]))
        nodes.append(onnx.helper.make_node("MatMul", [prev_features, "feedback_weights"], [fb_proj]))
        nodes.append(onnx.helper.make_node("Mul", [in_proj, "input_scaling"], [in_proj_scaled]))
        nodes.append(onnx.helper.make_node("Mul", [fb_proj, "feedback_scale"], [fb_proj_scaled]))
        nodes.append(onnx.helper.make_node("Add", [in_proj_scaled, fb_proj_scaled], [angles_sum1]))
        nodes.append(onnx.helper.make_node("Add", [angles_sum1, "input_bias"], [angles]))

        angles_div_pi = f"{prefix}_angles_div_pi"
        angles_tanh = f"{prefix}_angles_tanh"
        angles_bounded = f"{prefix}_angles_bounded"
        nodes.append(onnx.helper.make_node("Div", [angles, "pi"], [angles_div_pi]))
        nodes.append(onnx.helper.make_node("Tanh", [angles_div_pi], [angles_tanh]))
        nodes.append(onnx.helper.make_node("Mul", [angles_tanh, "pi"], [angles_bounded]))

        # Rotation d'encodage RY(angle_q) par qubit, angle dynamique -- seul
        # bloc du circuit qui ne peut pas etre precalcule (cf. train_global_model).
        for q in range(n_qubits):
            angle_q = f"{prefix}_angle_q{q}"
            half_q = f"{prefix}_half_q{q}"
            cos_q = f"{prefix}_cos_q{q}"
            sin_q = f"{prefix}_sin_q{q}"
            neg_sin_q = f"{prefix}_neg_sin_q{q}"
            gate_flat_q = f"{prefix}_gate_flat_q{q}"
            gate_q = f"{prefix}_gate_q{q}"
            new_state_real = f"{prefix}_state_real_q{q}"
            new_state_imag = f"{prefix}_state_imag_q{q}"

            nodes.append(onnx.helper.make_node("Gather", [angles_bounded, f"idx_q{q}"], [angle_q], axis=0))
            nodes.append(onnx.helper.make_node("Mul", [angle_q, "half"], [half_q]))
            nodes.append(onnx.helper.make_node("Cos", [half_q], [cos_q]))
            nodes.append(onnx.helper.make_node("Sin", [half_q], [sin_q]))
            nodes.append(onnx.helper.make_node("Mul", [sin_q, "neg_one"], [neg_sin_q]))
            nodes.append(onnx.helper.make_node("Concat", [cos_q, neg_sin_q, sin_q, cos_q], [gate_flat_q], axis=0))
            nodes.append(onnx.helper.make_node("Reshape", [gate_flat_q, "shape_2x2"], [gate_q]))

            _apply_qubit_gate_onnx_nodes(nodes, state_real, gate_q, q, n_qubits, shape_names, new_state_real)
            _apply_qubit_gate_onnx_nodes(nodes, state_imag, gate_q, q, n_qubits, shape_names, new_state_imag)
            state_real, state_imag = new_state_real, new_state_imag

        # Application de la matrice fixe precalculee (rotations reservoir_angles
        # + permutations CNOT) : U = U_real + i*U_imag, state = state_real + i*state_imag
        # state_out = U @ state => decomposition reelle (cf. section 5a) :
        #   out_real = U_real@state_real - U_imag@state_imag
        #   out_imag = U_real@state_imag + U_imag@state_real
        fx_rr = f"{prefix}_fx_rr"
        fx_ii = f"{prefix}_fx_ii"
        fx_ri = f"{prefix}_fx_ri"
        fx_ir = f"{prefix}_fx_ir"
        fixed_state_real = f"{prefix}_fixed_state_real"
        fixed_state_imag = f"{prefix}_fixed_state_imag"
        nodes.append(onnx.helper.make_node("MatMul", [state_real, "U_real_T"], [fx_rr]))
        nodes.append(onnx.helper.make_node("MatMul", [state_imag, "U_imag_T"], [fx_ii]))
        nodes.append(onnx.helper.make_node("MatMul", [state_real, "U_imag_T"], [fx_ri]))
        nodes.append(onnx.helper.make_node("MatMul", [state_imag, "U_real_T"], [fx_ir]))
        nodes.append(onnx.helper.make_node("Sub", [fx_rr, fx_ii], [fixed_state_real]))
        nodes.append(onnx.helper.make_node("Add", [fx_ri, fx_ir], [fixed_state_imag]))

        # Normalisation L2 (toujours appliquee, comme le chemin Python -- le
        # garde `if norm > 0` du numpy ne s'active jamais en pratique car
        # l'evolution est unitaire donc norm ~= 1 a chaque pas, cf. discussion).
        sq_real = f"{prefix}_sq_real"
        sq_imag = f"{prefix}_sq_imag"
        probs = f"{prefix}_probs"
        norm_sq = f"{prefix}_norm_sq"
        norm = f"{prefix}_norm"
        norm_state_real = f"{prefix}_norm_state_real"
        norm_state_imag = f"{prefix}_norm_state_imag"
        nodes.append(onnx.helper.make_node("Mul", [fixed_state_real, fixed_state_real], [sq_real]))
        nodes.append(onnx.helper.make_node("Mul", [fixed_state_imag, fixed_state_imag], [sq_imag]))
        nodes.append(onnx.helper.make_node("Add", [sq_real, sq_imag], [probs]))
        nodes.append(onnx.helper.make_node("ReduceSum", [probs], [norm_sq], keepdims=1))
        nodes.append(onnx.helper.make_node("Sqrt", [norm_sq], [norm]))
        nodes.append(onnx.helper.make_node("Div", [fixed_state_real, norm], [norm_state_real]))
        nodes.append(onnx.helper.make_node("Div", [fixed_state_imag, norm], [norm_state_imag]))
        state_real, state_imag = norm_state_real, norm_state_imag

        # Mesure Pauli-Z par qubit : measured_q = sum(probs_normalisees * z_signs[q])
        sq_real_n = f"{prefix}_sq_real_n"
        sq_imag_n = f"{prefix}_sq_imag_n"
        probs_n = f"{prefix}_probs_n"
        nodes.append(onnx.helper.make_node("Mul", [state_real, state_real], [sq_real_n]))
        nodes.append(onnx.helper.make_node("Mul", [state_imag, state_imag], [sq_imag_n]))
        nodes.append(onnx.helper.make_node("Add", [sq_real_n, sq_imag_n], [probs_n]))

        measured_parts = []
        for q in range(n_qubits):
            z_signs_q_name = f"z_signs_q{q}"
            if t == 0:
                initializers.append(
                    onnx.numpy_helper.from_array(np.asarray(reservoir["z_signs"][q], dtype=np.float32), z_signs_q_name)
                )
            weighted_q = f"{prefix}_weighted_q{q}"
            measured_q = f"{prefix}_measured_q{q}"
            nodes.append(onnx.helper.make_node("Mul", [probs_n, z_signs_q_name], [weighted_q]))
            nodes.append(onnx.helper.make_node("ReduceSum", [weighted_q], [measured_q], keepdims=1))
            measured_parts.append(measured_q)
        measured = f"{prefix}_measured"
        nodes.append(onnx.helper.make_node("Concat", measured_parts, [measured], axis=0))

        # features = (1-leak)*prev_features + leak*measured
        prev_scaled = f"{prefix}_prev_scaled"
        measured_scaled = f"{prefix}_measured_scaled"
        features = f"{prefix}_features"
        nodes.append(onnx.helper.make_node("Mul", [prev_features, "one_minus_leak"], [prev_scaled]))
        nodes.append(onnx.helper.make_node("Mul", [measured, "leak_rate"], [measured_scaled]))
        nodes.append(onnx.helper.make_node("Add", [prev_scaled, measured_scaled], [features]))

        prev_features = features
        kept_states.append(features)

    initializers.append(onnx.numpy_helper.from_array(np.asarray([2, 2], dtype=np.int64), "shape_2x2"))

    last_state = kept_states[-1]
    summary_inputs = [last_state]

    if state_summary in {"last_mean", "last_mean_std"}:
        unsqueezed = []
        for idx, state_name in enumerate(kept_states):
            out_name = f"seq_{idx}"
            nodes.append(onnx.helper.make_node("Reshape", [state_name, "shape_1_nqubits"], [out_name]))
            unsqueezed.append(out_name)
        initializers.append(onnx.numpy_helper.from_array(np.asarray([1, n_qubits], dtype=np.int64), "shape_1_nqubits"))
        nodes.append(onnx.helper.make_node("Concat", unsqueezed, ["states_seq"], axis=0))
        nodes.append(onnx.helper.make_node("ReduceMean", ["states_seq"], ["states_mean"], axes=[0], keepdims=0))
        summary_inputs.append("states_mean")

    if state_summary == "last_mean_std":
        nodes.extend(
            [
                onnx.helper.make_node("Sub", ["states_seq", "states_mean"], ["centered_states"]),
                onnx.helper.make_node("Mul", ["centered_states", "centered_states"], ["var_terms"]),
                onnx.helper.make_node("ReduceMean", ["var_terms"], ["states_var"], axes=[0], keepdims=0),
                onnx.helper.make_node("Sqrt", ["states_var"], ["states_std"]),
            ]
        )
        summary_inputs.append("states_std")
    elif state_summary not in {"last", "last_mean"}:
        raise ValueError(f"state_summary inconnu: {state_summary}")

    if len(summary_inputs) == 1:
        nodes.append(onnx.helper.make_node("Identity", [summary_inputs[0]], ["summary"]))
    else:
        nodes.append(onnx.helper.make_node("Concat", summary_inputs, ["summary"], axis=0))

    summary_dim = n_qubits * len(summary_inputs)
    initializers.append(onnx.numpy_helper.from_array(np.asarray([1, summary_dim], dtype=np.int64), "shape_summary_2d"))
    nodes.append(onnx.helper.make_node("Reshape", ["summary", "shape_summary_2d"], ["summary_2d"]))

    nodes.extend(
        [
            onnx.helper.make_node("Gemm", ["summary_2d", "Wout", "bias"], ["logits"], alpha=1.0, beta=1.0),
            onnx.helper.make_node("Softmax", ["logits"], ["probability_tensor"], axis=1),
            onnx.helper.make_node("ArgMax", ["logits"], ["label_tensor"], axis=1, keepdims=0),
        ]
    )

    num_classes = int(Wout.shape[1])
    graph = onnx.helper.make_graph(
        nodes,
        model_name,
        graph_inputs,
        [
            onnx.helper.make_tensor_value_info("probability_tensor", onnx.TensorProto.FLOAT, [1, num_classes]),
            onnx.helper.make_tensor_value_info("label_tensor", onnx.TensorProto.INT64, [1]),
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


def _predict_onnx_qrc(onnx_path, x_windows):
    """x_windows: array (n_windows, window_size, n_features), DEJA scale
    (meme convention que le chemin Python, cf. note dans _build_onnx_qrc_graph)."""
    ort_session = ort.InferenceSession(str(onnx_path))
    y_pred = np.zeros(len(x_windows), dtype=np.int32)
    for i, window in enumerate(x_windows):
        outputs = ort_session.run(None, {"x_window": window.astype(np.float32)})
        y_pred[i] = int(outputs[1][0])
    return y_pred


def _extract_raw_qrc_windows(df_sample, window_size, step_size):
    """
    Reconstruit les fenetres BRUTES (deja scale par le RobustScaler du fold,
    meme convention que extract_qrc_windows, mais PAS encore transformees par
    le reservoir) -- necessaire car extract_qrc_windows ne retourne que les
    features DEJA resumees par _transform_window, alors que le graphe ONNX
    prend en entree une fenetre brute (x_window). Reutilisee par
    _validate_onnx_qrc ET train_global_model (meme duplication assumee que
    _extract_raw_sensor_windows dans MESN.py, appelee separement aux deux
    endroits).
    """
    df_sorted = _sort_df(df_sample)
    groups = list(df_sorted.groupby([COL_PARTICIPANT, COL_SESSION], sort=False))

    x_raw_windows = []
    y_true = []
    for (participant, session), group in groups:
        x_raw = group[SIGNAL_COLS].values.astype(np.float32)
        y_raw = group[COL_LABEL].values.astype(np.int32)
        for start in range(0, len(x_raw) - window_size + 1, step_size):
            end = start + window_size
            x_raw_windows.append(x_raw[start:end])
            y_true.append(_window_label(y_raw[start:end]))

    return x_raw_windows, np.asarray(y_true, dtype=np.int32)


def _validate_onnx_qrc(onnx_model, reservoir, readout_model, df_sample, window_size, step_size):
    """
    Validation numerique OBLIGATOIRE du graphe ONNX QRC contre le chemin
    Python (_qrc_states + predict_readout), fenetre par fenetre (comme
    _validate_onnx_mesn) sur un echantillon de sessions. Exige un agreement
    >= 95%, rapporte le MAE -- ne garantit PAS l'export si le seuil n'est pas
    atteint (appelant responsable de decider quoi faire du resultat).
    """
    print("\n  --- Validation ONNX QRC ---")
    x_raw_windows, y_true = _extract_raw_qrc_windows(df_sample, window_size, step_size)

    if not x_raw_windows:
        print("  Validation ONNX QRC : aucune fenetre dans l'echantillon, validation ignoree.")
        return None, None

    x_features = np.asarray([_transform_window(w, reservoir) for w in x_raw_windows], dtype=np.float32)
    y_pred_python = predict_readout(readout_model, x_features)

    ort_session = ort.InferenceSession(onnx_model.SerializeToString())
    n = len(x_raw_windows)
    y_pred_onnx = np.zeros(n, dtype=np.int32)
    # model.run() renvoie les scores BRUTS de la regression Ridge (pre-softmax),
    # alors que le graphe ONNX expose probability_tensor DEJA softmax (comme
    # _ReadoutProbaWrapper.predict_proba dans MESN.py) -- softmax manuel ici
    # pour comparer des quantites homogenes (l'argmax, lui, est invariant au
    # softmax, donc y_pred_python/y_pred_onnx ne sont pas affectes par ceci).
    scores_python = np.asarray(readout_model.run(x_features), dtype=np.float32)
    if scores_python.ndim == 1:
        scores_python = scores_python.reshape(1, -1)
    exp_scores = np.exp(scores_python - scores_python.max(axis=1, keepdims=True))
    proba_python = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    proba_onnx = np.zeros_like(proba_python)

    for i, window in enumerate(x_raw_windows):
        outputs = ort_session.run(None, {"x_window": window.astype(np.float32)})
        proba_onnx[i] = outputs[0][0]
        y_pred_onnx[i] = int(outputs[1][0])

    agreement = float(np.mean(y_pred_python == y_pred_onnx)) * 100
    mae = float(np.mean(np.abs(proba_python - proba_onnx)))
    print(f"  Validation ONNX QRC : {agreement:.1f}% d'accord sur {n} fenetres, MAE={mae:.6f}")
    if agreement < 95.0:
        print(
            f"  WARNING : accord ONNX/Python ({agreement:.1f}%) sous le seuil de 95%. "
            f"Verifier la construction du graphe (rotation d'encodage par qubit, "
            f"application de fixed_layer_matrix, ordre du resume last/mean/std)."
        )

    return agreement, mae


# ============================================================================
# 6. MODELE GLOBAL
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
    config = _qrc_window_config()
    window_size, step_size = config["window_size"], config["step_size"]

    try:
        idx = np.random.default_rng(42).permutation(
            df_labeled[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().shape[0]
        )
        unique_sessions_all = [
            tuple(x) for x in df_labeled[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
        ]
        split_point = int(0.9 * len(unique_sessions_all))
        train_sessions = [unique_sessions_all[i] for i in idx[:split_point]]
        val_sessions = [unique_sessions_all[i] for i in idx[split_point:]]

        df_tr_raw = df_labeled[
            df_labeled.set_index([COL_PARTICIPANT, COL_SESSION]).index.isin(train_sessions)
        ].copy()
        df_vl_raw = df_labeled[
            df_labeled.set_index([COL_PARTICIPANT, COL_SESSION]).index.isin(val_sessions)
        ].copy()

        # SMOTE seulement sur le train, avant scaling et extraction de fenetres
        df_tr_raw = resample_dataframe(df_tr_raw, SIGNAL_COLS)

        scaler = RobustScaler()
        df_tr_raw[SIGNAL_COLS] = scaler.fit_transform(df_tr_raw[SIGNAL_COLS])
        df_vl_raw[SIGNAL_COLS] = scaler.transform(df_vl_raw[SIGNAL_COLS])

        reservoir = _build_qrc(best_params, len(SIGNAL_COLS))
        x_train, y_train = extract_qrc_windows(df_tr_raw, reservoir, window_size, step_size, tag="global_train", verbose=True)
        x_val, y_val = extract_qrc_windows(df_vl_raw, reservoir, window_size, step_size, tag="global_val", verbose=True)

        if len(y_train) == 0 or len(y_val) == 0:
            print("  Modele global echoue : train/val vide")
            return

        model = build_model(best_params)
        fit_readout(model, x_train, y_train)
        y_pred = predict_readout(model, x_val)

        f1_mac = f1_score(y_val, y_pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        print(f"  F1-Macro (val globale) : {f1_mac:.4f} | Bal.Acc : {bal_acc:.4f}")

        models_dir = MODELS_DIR / "QRC"
        models_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = models_dir / f"{MODEL_NAME}_global.pkl"
        joblib.dump({"reservoir": reservoir, "readout": model, "scaler": scaler}, pkl_path)
        print(f"  Pickle backup : {pkl_path}")

        edge_metrics = {
            "native_qrc": _compute_classification_metrics(y_val, y_pred)
        }
        edge_metrics["native_qrc"]["model_size_kb"] = round(pkl_path.stat().st_size / 1024, 2)

        # Export ONNX -- tentative reelle (pas une hypothese codee en dur) :
        # rendue possible par fixed_layer_matrix (constante precalculee, plus
        # de boucle de gates a derouler pour la partie fixe du circuit). Seule
        # la rotation d'encodage d'entree (dependante de x_t/prev_features)
        # est construite dynamiquement dans le graphe. Echec documente avec le
        # type d'exception si une etape quelconque (construction, validation,
        # ecriture) echoue -- ne bloque pas le reste de train_global_model().
        try:
            onnx_model = _build_onnx_qrc_graph(reservoir, model, f"{MODEL_NAME}_global")
            agreement, mae = _validate_onnx_qrc(onnx_model, reservoir, model, df_vl_raw, window_size, step_size)

            if agreement is None:
                edge_metrics["onnx_export_status"] = "echec : validation impossible (aucune fenetre dans l'echantillon)"
            elif agreement < 95.0:
                edge_metrics["onnx_export_status"] = (
                    f"echec : accord ONNX/Python {agreement:.1f}% sous le seuil de 95% (MAE={mae:.6f}) "
                    f"-- graphe non retenu comme resultat final, cf. logs pour diagnostic."
                )
            else:
                onnx_path = models_dir / f"{MODEL_NAME}_global.onnx"
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                print(f"  ONNX exporte : {onnx_path}")

                x_val_raw, y_val_onnx = _extract_raw_qrc_windows(df_vl_raw, window_size, step_size)
                if x_val_raw and len(y_val_onnx) == len(y_val) and np.array_equal(y_val_onnx, y_val):
                    y_pred_onnx = _predict_onnx_qrc(onnx_path, x_val_raw)
                    edge_metrics["onnx"] = _compute_classification_metrics(y_val_onnx, y_pred_onnx)
                    edge_metrics["onnx"]["model_size_kb"] = round(onnx_path.stat().st_size / 1024, 2)
                    edge_metrics["onnx"]["validation_agreement_pct"] = round(agreement, 2)
                    edge_metrics["onnx"]["validation_mae"] = round(mae, 6)
                    edge_metrics["onnx"]["stm32_flash_budget_kb"] = STM32_FLASH_KB
                    edge_metrics["onnx"]["fits_flash_budget"] = bool(
                        edge_metrics["onnx"]["model_size_kb"] <= STM32_FLASH_KB
                    )
                else:
                    edge_metrics["onnx_export_status"] = (
                        "echec : desalignement entre les fenetres brutes reconstruites "
                        f"({len(y_val_onnx)}) et x_val/y_val ({len(y_val)}) -- metriques onnx non calculees."
                    )

        except Exception as exc:
            print(f"  Export ONNX QRC echoue ({type(exc).__name__}: {exc})")
            traceback.print_exc()
            edge_metrics["onnx_export_status"] = f"echec : {type(exc).__name__}: {exc}"

        _save_edge_comparison_metrics(MODEL_NAME, edge_metrics)

    except Exception as exc:
        print(f"  Modele global echoue ({type(exc).__name__}: {exc})")
        traceback.print_exc()


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
            # SMOTE seulement sur df_train, avant le scaling et l'extraction de fenetres
            df_train = resample_dataframe(df_train, SIGNAL_COLS)

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
