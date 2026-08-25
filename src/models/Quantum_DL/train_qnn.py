import gc
import json
import os
import sys
import warnings
from pathlib import Path
import tensorflow as tf
import seaborn as sns

# ── Limiter le parallélisme CPU (laisser la mémoire pour les VQC) ────
# PennyLane lightning.qubit est CPU-only et run_eagerly=True supprime tout
# parallélisme GPU. Forcer CPU évite l'OOM sur GPU limité sans aucune perte
# de performance — le goulot d'étranglement est le circuit quantique Python.
tf.config.set_visible_devices([], "GPU")

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")

# ══════════════════════════════════════════════════════════════════════
# 1. IMPORTS
# ══════════════════════════════════════════════════════════════════════
# pip install pennylane pennylane-lightning

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pennylane as qml

tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import *

from utils.apply_smote import resample_dataframe

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TerminateOnNaN

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# 2. CONSTANTES LOCALES
# ══════════════════════════════════════════════════════════════════════
MODEL_NAME    = "QNN"
LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES  = ["baseline", "activity", "pre_fatigue", "fatigue"]

# Entrée = vecteur de features (mean/std) résumant la fenêtre, pas une
# séquence : on réutilise le fenêtrage standard du projet (pas de fenêtre
# réduite ad hoc comme QLSTM, le VQC n'est ici évalué qu'une seule fois par
# fenêtre, pas une fois par timestep).
QNN_WINDOW_SIZE = WINDOW_CONFIGS["default"]["window_size"]
QNN_STEP_SIZE   = WINDOW_CONFIGS["default"]["step_size"]


# ══════════════════════════════════════════════════════════════════════
# 3. CIRCUIT QUANTIQUE (PennyLane) — appliqué UNE SEULE FOIS par fenêtre
# ══════════════════════════════════════════════════════════════════════

class VQCLayer(tf.keras.layers.Layer):
    """
    Couche quantique variationnelle (VQC) compatible Keras.
    Design identique à celui validé dans QLSTM.py :
      - device  : lightning.qubit (simulateur PennyLane rapide, CPU-only)
      - diff    : "adjoint" — gradient via différentiation adjointe
      - batch   : boucle Python native (pas tf.map_fn, incompatible avec
                  l'AutoGraph interne de PennyLane en mode eager)
      - poids   : add_weight float32 (Keras standard),
                  cast → float64 à l'entrée du circuit (requis par PennyLane)
                  cast → float32 en sortie (compatible reste du modèle)

    Circuit :
      1. AngleEmbedding(RY)        — angle ∈ [-π,π] → rotation sur sphère de Bloch
      2. StronglyEntanglingLayers  — Rot(α,β,γ) + CNOT → entanglement inter-qubits
      3. mesure ⟨PauliZ⟩           — expectation value ∈ [-1, 1]

    Contrairement à QLSTM (VQC appelé pas-à-pas dans une cellule RNN) et à
    QRC/QKERNEL (poids quantiques fixes/aléatoires, pas de VQC entraîné), ici
    la couche est appelée UNE SEULE FOIS sur le vecteur de features de la
    fenêtre, et ses poids (`vqc_weights`) sont entraînés de bout en bout par
    rétropropagation comme n'importe quelle couche Keras.
    """

    def __init__(self, n_qubits: int, n_vqc_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits     = n_qubits
        self.n_vqc_layers = n_vqc_layers

        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="tf", diff_method="adjoint")
        def _circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self._circuit = _circuit

    def build(self, _input_shape):
        self.vqc_weights = self.add_weight(
            name="vqc_weights",
            shape=(self.n_vqc_layers, self.n_qubits, 3),
            initializer=tf.keras.initializers.RandomUniform(minval=-np.pi, maxval=np.pi),
            trainable=True,
        )
        super().build(_input_shape)

    def call(self, inputs):
        """inputs : (batch, n_qubits) float32 → (batch, n_qubits) float32"""
        # Boucle Python native (invisible à AutoGraph en mode run_eagerly=True) :
        # cf. QLSTM.py pour le détail de l'incompatibilité tf.map_fn/PennyLane.
        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = int(tf.shape(inputs)[0])
        w64 = tf.cast(self.vqc_weights, tf.float64)  # déplacé hors de la boucle
        results = []
        for i in range(int(batch_size)):
            x64 = tf.cast(inputs[i], tf.float64)
            results.append(tf.cast(tf.stack(self._circuit(x64, w64)), tf.float32))
        return tf.stack(results, axis=0)

    def compute_output_shape(self, input_shape):
        # Requis en Keras 3 : utilisé directement dans la Functional API (pas
        # dans une RNN qui fournirait output_size), Keras a besoin de cette
        # méthode pour inférer la forme symbolique sans exécuter call() sur un
        # KerasTensor (call() fait `int(tf.shape(inputs)[0])`, invalide sur un
        # tenseur symbolique).
        return input_shape[:-1] + (self.n_qubits,)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_qubits": self.n_qubits, "n_vqc_layers": self.n_vqc_layers})
        return cfg


class VQCOutputScale(tf.keras.layers.Layer):
    """
    Facteur d'échelle scalaire entraînable appliqué à la sortie du VQC
    (déjà bornée [-1,1] par la mesure PauliZ) avant la tête Dense.

    Implémenté via `add_weight` (couche Keras dédiée) plutôt qu'un
    `tf.Variable` posé directement dans le graphe Functional, pour garantir
    que le poids apparaît bien dans `model.trainable_variables` — vérifié
    explicitement par le test minimal en bas de ce fichier.
    """

    def build(self, _input_shape):
        self.scale = self.add_weight(
            name="vqc_output_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
        )
        super().build(_input_shape)

    def call(self, inputs):
        return inputs * self.scale


# ══════════════════════════════════════════════════════════════════════
# 4. MÉMOIRE
# ══════════════════════════════════════════════════════════════════════

def _free_memory(model=None):
    if model is not None:
        del model
    tf.keras.backend.clear_session()
    gc.collect()


# ══════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ══════════════════════════════════════════════════════════════════════

def plot_fold(history, fold_idx, test_part, test_sess, save_dir,
              f1_mac, bal_acc, y_true=None, y_pred=None):
    epochs_range = range(1, len(history.history["loss"]) + 1)
    best_epoch   = int(np.argmin(history.history["val_loss"])) + 1

    has_cm = (y_true is not None) and (y_pred is not None)
    ncols  = 3 if has_cm else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    fig.suptitle(
        f"QNN  |  Fold {fold_idx} — P{test_part} S{test_sess}"
        f"  |  F1-Macro: {f1_mac:.3f}  |  Bal.Acc: {bal_acc:.3f}",
        fontsize=13, fontweight="bold",
    )

    axes[0].plot(epochs_range, history.history["loss"],     label="Train Loss", color="#2196F3", linewidth=2)
    axes[0].plot(epochs_range, history.history["val_loss"], label="Val Loss",   color="#F44336", linewidth=2, linestyle="--")
    axes[0].axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5, label=f"Best ({best_epoch})")
    axes[0].set_title("Loss — Train vs Validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history.history["accuracy"],     label="Train Acc", color="#4CAF50", linewidth=2)
    axes[1].plot(epochs_range, history.history["val_accuracy"], label="Val Acc",   color="#FF9800", linewidth=2, linestyle="--")
    axes[1].axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5, label=f"Best ({best_epoch})")
    axes[1].set_title("Accuracy — Train vs Validation")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    if has_cm:
        cm     = confusion_matrix(y_true, y_pred)
        labels = [TARGET_NAMES[i] for i in np.unique(np.concatenate([y_true, y_pred]))]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[2], linewidths=0.5, linecolor="gray")
        axes[2].set_title("Confusion Matrix")
        axes[2].set_xlabel("Prédit"); axes[2].set_ylabel("Réel")
        axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plot_path = save_dir / f"fold{fold_idx:02d}_P{test_part}_S{test_sess}_curves.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Courbes sauvegardées : {plot_path}")


# ══════════════════════════════════════════════════════════════════════
# 6. DONNÉES — vecteurs de features (mean/std), pas de tf.data.Dataset
# ══════════════════════════════════════════════════════════════════════

def _window_label(y_window):
    """Vote majoritaire sur les labels d'une fenêtre — même logique que QRC.py."""
    values, counts = np.unique(y_window, return_counts=True)
    return int(values[np.argmax(counts)])


def extract_window_features(df, window_size, step_size):
    """Features statistiques par fenetre (mean/std par colonne de signal)."""
    x_all, y_all = [], []
    df_sorted = (
        df.sort_values([COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP])
        if COL_TIMESTAMP in df.columns
        else df.sort_values([COL_PARTICIPANT, COL_SESSION])
    )
    for (participant, session), group in df_sorted.groupby([COL_PARTICIPANT, COL_SESSION], sort=False):
        x_raw = group[SIGNAL_COLS].values.astype(np.float32)
        y_raw = group[COL_LABEL].values.astype(np.int32)
        for start in range(0, len(x_raw) - window_size + 1, step_size):
            end = start + window_size
            window = x_raw[start:end]
            feat = np.concatenate([window.mean(axis=0), window.std(axis=0)])
            x_all.append(feat)
            y_all.append(_window_label(y_raw[start:end]))
    if not x_all:
        return np.empty((0, 2 * len(SIGNAL_COLS)), dtype=np.float32), np.array([], dtype=np.int32)
    return np.asarray(x_all, dtype=np.float32), np.asarray(y_all, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════
# 7. CONSTRUCTION DU MODÈLE
# ══════════════════════════════════════════════════════════════════════

def build_model(input_shape, num_classes, params):
    """
    Architecture complète :

        Input (n_features,)             ← features mean/std de la fenêtre
          │
         Dense(n_qubits)                 ← projection classique
          │
         tanh × π                        ← bornage de l'angle d'encodage
          │
         [VQCLayer]                      ← VQC entraîné, UNE SEULE évaluation
          │
         × vqc_output_scale (entraînable)
          │
         Dense(dense_units, relu)        ← tête classique à poids appris
          │
         Dropout
          │
         Dense(num_classes, softmax)

    Contrairement à QLSTM, pas de `tf.keras.layers.RNN` : le VQC est appelé
    une seule fois sur le vecteur de features de la fenêtre, pas pas à pas.
    """
    n_qubits     = params["n_qubits"]
    n_vqc_layers = params["n_vqc_layers"]
    dense_units  = params.get("dense_units", 64)
    dropout_rate = params.get("dropout_rate", 0.3)
    l2_strength  = params.get("l2_reg", 0.0)
    lr           = params["learning_rate"]
    reg = l2(l2_strength) if l2_strength > 0 else None

    inp = Input(shape=input_shape)  # (n_features,) -- mean/std, pas de sequence
    proj = Dense(n_qubits, use_bias=True, name="proj_to_angles")(inp)
    # Keras 3 (Functional API stricte) interdit les ops tf.* brutes sur un
    # KerasTensor hors d'une Layer -> Lambda, pas d'appel direct a tf.math.tanh.
    angles = tf.keras.layers.Lambda(
        lambda t: tf.math.tanh(t) * np.pi, name="angle_bound"
    )(proj)  # borne dans (-pi, pi), applique des le depart

    vqc_out = VQCLayer(n_qubits, n_vqc_layers, name="vqc")(angles)  # [-1, 1], UNE seule evaluation
    vqc_out_scaled = VQCOutputScale(name="vqc_output_scale_layer")(vqc_out)

    x = Dense(dense_units, activation="relu", kernel_regularizer=reg)(vqc_out_scaled)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,  # requis pour PennyLane, comme QLSTM
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# 8. OPTUNA
# ══════════════════════════════════════════════════════════════════════

def _prepare_split(df_train_raw, df_val_raw):
    """SMOTE sur train uniquement, RobustScaler fit sur train, puis
    extraction des features (mean/std) -- les features ne dependent pas des
    hyperparametres du modele, donc precalculees UNE FOIS par split."""
    df_train = resample_dataframe(df_train_raw, SIGNAL_COLS)
    df_val   = df_val_raw.copy()

    scaler = RobustScaler()
    df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
    df_val[SIGNAL_COLS]   = scaler.transform(df_val[SIGNAL_COLS])

    X_train, y_train = extract_window_features(df_train, QNN_WINDOW_SIZE, QNN_STEP_SIZE)
    X_val,   y_val   = extract_window_features(df_val,   QNN_WINDOW_SIZE, QNN_STEP_SIZE)
    return X_train, y_train, X_val, y_val


def optuna_objective(trial, feature_splits, num_classes):
    n_qubits      = trial.suggest_categorical("n_qubits",      QNN_OPTUNA_SPACE["n_qubits"])
    n_vqc_layers  = trial.suggest_categorical("n_vqc_layers",  QNN_OPTUNA_SPACE["n_vqc_layers"])
    dense_units   = trial.suggest_categorical("dense_units",   QNN_OPTUNA_SPACE["dense_units"])
    dropout_rate  = trial.suggest_float("dropout_rate", **QNN_OPTUNA_SPACE["dropout_rate"])
    l2_reg        = trial.suggest_float("l2_reg",        **QNN_OPTUNA_SPACE["l2_reg"])
    learning_rate = trial.suggest_float("learning_rate", **QNN_OPTUNA_SPACE["learning_rate"])
    batch_size    = trial.suggest_categorical("batch_size", QNN_OPTUNA_SPACE["batch_size"])

    params = {
        "n_qubits": n_qubits, "n_vqc_layers": n_vqc_layers,
        "dense_units": dense_units, "dropout_rate": dropout_rate,
        "l2_reg": l2_reg, "learning_rate": learning_rate, "batch_size": batch_size,
    }
    scores = []

    for (X_train, y_train, X_val, y_val) in feature_splits:
        if len(y_train) == 0 or len(y_val) == 0:
            scores.append(0.0)
            continue
        model = None
        try:
            model = build_model((X_train.shape[1],), num_classes, params)
            y_train_oh = to_categorical(y_train, num_classes=num_classes)
            y_val_oh   = to_categorical(y_val,   num_classes=num_classes)
            model.fit(
                X_train, y_train_oh,
                validation_data=(X_val, y_val_oh),
                epochs=QNN_OPTUNA_EPOCHS,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                    TerminateOnNaN(),
                ],
                verbose=0,
            )
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        except tf.errors.ResourceExhaustedError:
            raise optuna.exceptions.TrialPruned("OOM")
        except Exception as exc:
            print(f"  [WARN] split échoué ({type(exc).__name__}: {exc})")
            scores.append(0.0)
        finally:
            _free_memory(model)
            gc.collect()

    return float(np.mean(scores)) if scores else 0.0


def optimize_hyperparams(df, num_classes):
    n_trials = QNN_OPTUNA_TRIALS
    n_sess   = QNN_OPTUNA_SESSIONS
    print(f"\nOPTUNA {MODEL_NAME} — {n_trials} trials | {n_sess} sessions\n" + "=" * 60)
    import random
    random.seed(42)

    unique_sessions = [tuple(x) for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
    val_sessions    = random.sample(unique_sessions, min(n_sess, len(unique_sessions)))
    print(f"Sessions Optuna : {val_sessions}")

    feature_splits = []
    for (val_part, val_sess) in val_sessions:
        df_train_raw = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
        df_val_raw   = df[ (df[COL_PARTICIPANT] == val_part)  & (df[COL_SESSION] == val_sess)].copy()
        feature_splits.append(_prepare_split(df_train_raw, df_val_raw))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, feature_splits, num_classes),
        n_trials=n_trials, show_progress_bar=True,
        n_jobs=1, gc_after_trial=True, catch=(Exception,),
    )
    del feature_splits; gc.collect()

    print(f"\nBest F1-Macro : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    OPTUNA_PATH_QNN.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_QNN, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return study.best_params, val_sessions


# ══════════════════════════════════════════════════════════════════════
# 9. MODÈLE GLOBAL
# ══════════════════════════════════════════════════════════════════════

def train_global_model(df, best_params, num_classes, val_sessions):
    print("\n" + "=" * 60 + f"\nMODÈLE GLOBAL — {MODEL_NAME}\n" + "=" * 60)

    val_sessions_set = set(val_sessions)
    mask_val = df.set_index([COL_PARTICIPANT, COL_SESSION]).index.isin(val_sessions_set)
    df_train = df[~mask_val].copy()
    df_val   = df[mask_val].copy()

    # SMOTE seulement sur df_train, avant le scaling
    df_train = resample_dataframe(df_train, SIGNAL_COLS)

    scaler = RobustScaler()
    df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
    df_val[SIGNAL_COLS]   = scaler.transform(df_val[SIGNAL_COLS])

    X_train, y_train = extract_window_features(df_train, QNN_WINDOW_SIZE, QNN_STEP_SIZE)
    X_val,   y_val   = extract_window_features(df_val,   QNN_WINDOW_SIZE, QNN_STEP_SIZE)

    if len(y_train) == 0:
        print("  Modèle global échoué : train vide")
        return

    w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    bs = best_params.get("batch_size", 16)

    model = None
    try:
        model = build_model((X_train.shape[1],), num_classes, best_params)
        y_train_oh = to_categorical(y_train, num_classes=num_classes)
        has_val    = len(y_val) > 0
        y_val_oh   = to_categorical(y_val, num_classes=num_classes) if has_val else None

        monitor   = "val_loss" if has_val else "loss"
        callbacks = [
            EarlyStopping(monitor=monitor, patience=10, restore_best_weights=True),
            TerminateOnNaN(),
        ]
        model.fit(
            X_train, y_train_oh,
            validation_data=(X_val, y_val_oh) if has_val else None,
            epochs=LOSO_EPOCHS_QNN,
            batch_size=bs,
            shuffle=True,
            callbacks=callbacks,
            class_weight=dict(zip(np.unique(y_train), w)),
            verbose=1,
        )

        models_dir = MODELS_DIR / "QNN"
        models_dir.mkdir(parents=True, exist_ok=True)
        keras_path = models_dir / f"{MODEL_NAME}_global.keras"
        model.save(keras_path)
        print(f"  Modele natif sauvegarde : {keras_path}")

        if has_val:
            y_pred_val = np.argmax(model.predict(X_val, verbose=0), axis=1)
            edge_metrics = {
                "native_qnn": {
                    "f1_macro": float(f1_score(y_val, y_pred_val, average="macro", zero_division=0)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred_val)),
                    "model_size_kb": round(keras_path.stat().st_size / 1024, 2),
                    "deployment_status": (
                        "export_stm32_non_tente : contrairement a QLSTM, ce VQC n'est evalue "
                        "qu'UNE FOIS par fenetre (pas de recurrence), donc structurellement "
                        "plus simple a exporter en principe (un seul appel de circuit a "
                        "traduire, comme QRC, plutot que 4 VQC x window_size appels). "
                        "Non tente dans cette premiere passe -- a explorer en priorite sur "
                        "QNN plutot que QLSTM si les resultats F1 le justifient, precisement "
                        "grace a cette simplicite structurelle."
                    ),
                }
            }
            curr_metrics = {}
            if METRICS_PATH.exists():
                try:
                    with open(METRICS_PATH, "r") as f:
                        content = f.read().strip()
                    curr_metrics = json.loads(content) if content else {}
                except (json.JSONDecodeError, ValueError):
                    curr_metrics = {}
            curr_metrics.setdefault(MODEL_NAME, {})
            curr_metrics[MODEL_NAME]["edge_comparison"] = edge_metrics
            with open(METRICS_PATH, "w") as f:
                json.dump(curr_metrics, f, indent=4)
            print(f"  Comparaison edge sauvegardee : {METRICS_PATH}")
        else:
            print("  Pas de val disponible, edge_comparison non calculee.")
    except Exception as exc:
        print(f"  Modèle global échoué ({type(exc).__name__}: {exc})")
    finally:
        _free_memory(model)
        del df_train, df_val
        gc.collect()


# ══════════════════════════════════════════════════════════════════════
# 10. BOUCLE LOSO
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60 + f"\nTRAIN LOSO — {MODEL_NAME}\n" + "=" * 60)

    if not DATA_MODEL_READY.exists():
        return print(f"Dataset non trouvé : {DATA_MODEL_READY}")

    df = pd.read_csv(DATA_MODEL_READY)
    df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors="coerce").fillna(-1).astype(int)
    n_unlabeled = int((df[COL_LABEL] == -1).sum())
    df = df[df[COL_LABEL] >= 0].copy()
    print(f"Labeled: {len(df)} lignes | Unlabeled exclues: {n_unlabeled} lignes")

    unique_sessions = [
        tuple(x) for x in
        df[df[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
    ]
    num_classes = len(LABEL_MAPPING)

    if USE_OPTUNA_QNN:
        best_params, val_sessions = optimize_hyperparams(df, num_classes)
    else:
        best_params = MODEL_PARAMS["QNN"]
        import random as _rnd
        _rnd.seed(42)
        all_sess     = [tuple(x) for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
        val_sessions = _rnd.sample(all_sess, min(QNN_OPTUNA_SESSIONS, len(all_sess)))

    all_metrics = []
    import random
    random.seed(42)

    for test_idx, (test_part, test_sess) in enumerate(unique_sessions):
        print(f"\n--- FOLD {test_idx+1}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")

        df_pool = df[~((df[COL_PARTICIPANT] == test_part) & (df[COL_SESSION] == test_sess))].copy()
        df_test = df[ (df[COL_PARTICIPANT] == test_part)  & (df[COL_SESSION] == test_sess)].copy()

        pool_sessions      = [tuple(x) for x in df_pool[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
        val_part, val_sess = random.choice(pool_sessions)
        print(f"  Val : P{val_part} S{val_sess}")

        df_fit = df_pool[~((df_pool[COL_PARTICIPANT] == val_part) & (df_pool[COL_SESSION] == val_sess))].copy()
        df_val = df_pool[ (df_pool[COL_PARTICIPANT] == val_part)  & (df_pool[COL_SESSION] == val_sess)].copy()

        # SMOTE seulement sur df_fit, avant le scaling
        df_fit = resample_dataframe(df_fit, SIGNAL_COLS)

        scaler = RobustScaler()
        df_fit[SIGNAL_COLS]  = scaler.fit_transform(df_fit[SIGNAL_COLS])
        df_val[SIGNAL_COLS]  = scaler.transform(df_val[SIGNAL_COLS])
        df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])

        X_fit,  y_fit  = extract_window_features(df_fit,  QNN_WINDOW_SIZE, QNN_STEP_SIZE)
        X_val,  y_val  = extract_window_features(df_val,  QNN_WINDOW_SIZE, QNN_STEP_SIZE)
        X_test, y_test = extract_window_features(df_test, QNN_WINDOW_SIZE, QNN_STEP_SIZE)

        if len(y_test) == 0:
            print(f"  WARNING: test vide pour P{test_part}. Skip."); continue
        if len(y_val) == 0 or len(y_fit) == 0:
            print(f"  WARNING: Fit/Val vide. Skip."); continue

        w = compute_class_weight("balanced", classes=np.unique(y_fit), y=y_fit)
        bs = best_params.get("batch_size", 16)

        model = None
        try:
            model = build_model((X_fit.shape[1],), num_classes, best_params)
            y_fit_oh  = to_categorical(y_fit, num_classes=num_classes)
            y_val_oh  = to_categorical(y_val, num_classes=num_classes)
            history = model.fit(
                X_fit, y_fit_oh,
                validation_data=(X_val, y_val_oh),
                epochs=LOSO_EPOCHS_QNN,
                batch_size=bs,
                shuffle=True,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                    TerminateOnNaN(),
                ],
                class_weight=dict(zip(np.unique(y_fit), w)),
                verbose=1,
            )

            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

            f1_mac  = f1_score(y_test, y_pred, average="macro")
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            plots_dir = BASE_DIR / "training_curves" / "QNN"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_fold(history, test_idx + 1, test_part, test_sess,
                      plots_dir, f1_mac, bal_acc, y_test, y_pred)

            all_metrics.append({
                "Fold": test_idx + 1, "Participant": int(test_part), "Session": int(test_sess),
                "F1_Macro": float(f1_mac), "Balanced_Accuracy": float(bal_acc),
            })

        except Exception as exc:
            print(f"  Fold {test_idx+1} échoué ({type(exc).__name__}: {exc})")
        finally:
            _free_memory(model)
            del df_fit, df_val, df_test, df_pool
            gc.collect()

    if not all_metrics:
        return print("Aucun fold complété.")

    df_res = pd.DataFrame(all_metrics)
    print("\n" + "=" * 60 + f"\nRÉSULTATS FINAUX — {MODEL_NAME}\n" + "=" * 60)
    print(df_res.describe().loc[["mean", "std"]])

    curr_metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                content = f.read().strip()
            curr_metrics = json.loads(content) if content else {}
        except (json.JSONDecodeError, ValueError):
            curr_metrics = {}

    curr_metrics[MODEL_NAME] = {
        "mean_f1":      float(df_res["F1_Macro"].mean()),
        "std_f1":       float(df_res["F1_Macro"].std()),
        "mean_bal_acc": float(df_res["Balanced_Accuracy"].mean()),
        "std_bal_acc":  float(df_res["Balanced_Accuracy"].std()),
        "params":       best_params,
        "folds":        all_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(curr_metrics, f, indent=4)
    print(f"\nMétriques → {METRICS_PATH}")

    train_global_model(df, best_params, num_classes, val_sessions)


if __name__ == "__main__":
    main()
