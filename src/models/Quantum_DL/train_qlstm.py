import gc
import json
import os
import sys
import warnings
from pathlib import Path
import tensorflow as tf
import seaborn as sns

# ── Limiter le parallélisme CPU (laisser la mémoire pour les VQC) ────
# PennyLane default.qubit est CPU-only et run_eagerly=True supprime tout
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
# pip install pennylane-lightning[gpu]   # optionnel : accélération CUDA RTX

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
MODEL_NAME        = "QLSTM"
LABEL_MAPPING     = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES      = ["baseline", "activity", "pre_fatigue", "fatigue"]

# Fenêtre réduite vs modèles classiques : le RNN se déroule pas-à-pas,
# chaque timestep appelle 4 VQCs → coût proportionnel à window_size.
QLSTM_WINDOW_SIZE   = 120
QLSTM_STEP_SIZE     = 60
QLSTM_OPTUNA_EPOCHS = 5     # epochs courts pendant Optuna (VQC = lent)
LOSO_EPOCHS         = 100   # EarlyStopping coupe bien avant pour le LOSO complet

# ══════════════════════════════════════════════════════════════════════
# 3. CIRCUITS QUANTIQUES (PennyLane)
# ══════════════════════════════════════════════════════════════════════

class VQCLayer(tf.keras.layers.Layer):
    """
    Couche quantique variationnelle (VQC) compatible Keras.

    Design validé pour PennyLane >= 0.36 (KerasLayer supprimé) :
      - device  : default.qubit (simulateur TF-natif)
      - diff    : "backprop" — gradient via TF autodiff, float32 partout
      - batch   : tf.map_fn sur la dimension batch
      - poids   : add_weight float32 (Keras standard)
                  cast → float64 à l'entrée du circuit (requis par default.qubit)
                  cast → float32 en sortie (compatible reste du modèle)

    Circuit :
      1. AngleEmbedding(RY)        — angle ∈ [0,π] → rotation sur sphère de Bloch
      2. StronglyEntanglingLayers  — Rot(α,β,γ) + CNOT → entanglement inter-qubits
      3. mesure ⟨PauliZ⟩           — expectation value ∈ [-1, 1]

    Note : lightning.qubit (adjoint) est plus rapide pour de grands circuits
    mais incompatible avec tf.map_fn sur PennyLane >= 0.36 (interface TF dépréciée).
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
        # tf.map_fn trace la fonction via AutoGraph, ce qui brise le code Python
        # interne de PennyLane (if/else sans return dans l'else). Une boucle
        # Python native est invisible à AutoGraph en mode run_eagerly=True.
        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = int(tf.shape(inputs)[0])
        w64 = tf.cast(self.vqc_weights, tf.float64)  # deplace hors de la boucle
        results = []
        for i in range(int(batch_size)):
            x64 = tf.cast(inputs[i], tf.float64)
            results.append(tf.cast(tf.stack(self._circuit(x64, w64)), tf.float32))
        return tf.stack(results, axis=0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_qubits": self.n_qubits, "n_vqc_layers": self.n_vqc_layers})
        return cfg


# ══════════════════════════════════════════════════════════════════════
# 4. CELLULE QLSTM
# ══════════════════════════════════════════════════════════════════════

class QLSTMCell(tf.keras.layers.Layer):
    """
    Cellule QLSTM hybride (Chen et al., 2022 "Quantum Long Short-Term Memory").

    Chaque gate LSTM classique est remplacé par un VQC indépendant :

        combined = [x_t ‖ h_{t-1}]               # concat (batch, n_feat + n_qubits)
        angles_g = proj_g(combined) ∈ [-π/2, π/2] # projection Dense classique
        gate_g   = VQC_g(angles_g)  ∈ [-1, 1]     # mesure expectation PauliZ
        g_t      = tanh(gate_g)                    # activation finale

        c_t = σ(f_gate) ⊙ c_{t-1} + σ(i_gate) ⊙ tanh(g_gate)
        h_t = σ(o_gate) ⊙ tanh(c_t)

    Pourquoi 4 VQCs séparés :
      - Poids quantiques indépendants par gate (max expressivité)
      - Fidèle au papier original Chen et al.

    Pourquoi proj classique avant VQC :
      - AngleEmbedding attend exactement n_qubits angles
      - L'espace d'entrée [x_t, h_{t-1}] a n_feat+n_qubits dimensions
      - La Dense projette vers n_qubits et apprend quelle info encode
    """

    def __init__(self, n_qubits: int, n_vqc_layers: int, n_features: int, gate_scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits     = n_qubits
        self.n_vqc_layers = n_vqc_layers
        self.n_features   = n_features
        self.gate_scale   = float(gate_scale)

        # 4 VQCLayer indépendants — poids quantiques séparés par gate
        # Chaque VQCLayer crée son propre device default.qubit en interne
        self.qkl_i = VQCLayer(n_qubits, n_vqc_layers, name="vqc_i")
        self.qkl_f = VQCLayer(n_qubits, n_vqc_layers, name="vqc_f")
        self.qkl_g = VQCLayer(n_qubits, n_vqc_layers, name="vqc_g")
        self.qkl_o = VQCLayer(n_qubits, n_vqc_layers, name="vqc_o")

        # Projections classiques : [x_t, h_{t-1}] (n_feat+n_qubits) → n_qubits angles
        self.proj_i = Dense(n_qubits, use_bias=True, name="proj_i")
        self.proj_f = Dense(n_qubits, use_bias=True, name="proj_f")
        self.proj_g = Dense(n_qubits, use_bias=True, name="proj_g")
        self.proj_o = Dense(n_qubits, use_bias=True, name="proj_o")

        # Build immédiat avec forme connue (évite lazy build incompatible avec RNN)
        dummy_c = tf.zeros((1, n_features + n_qubits))
        dummy_q = tf.zeros((1, n_qubits))
        for proj in (self.proj_i, self.proj_f, self.proj_g, self.proj_o):
            proj(dummy_c)
        for vqc in (self.qkl_i, self.qkl_f, self.qkl_g, self.qkl_o):
            vqc(dummy_q)

    # ── Interface RNN cell ─────────────────────────────────────────────

    @property
    def state_size(self):
        return [self.n_qubits, self.n_qubits]   # [h_t, c_t]

    @property
    def output_size(self):
        return self.n_qubits  

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or tf.float32
        return [
            tf.zeros((batch_size, self.n_qubits), dtype=dtype),
            tf.zeros((batch_size, self.n_qubits), dtype=dtype),
        ]

    def build(self, _input_shape):
        super().build(_input_shape)

    def call(self, x_t, states):
        """
        x_t    : (batch, n_features) — entrée au temps t
        states : [h_{t-1}, c_{t-1}] — état précédent
        """
        h_prev, c_prev = states[0], states[1]

        # Concaténation entrée + état caché précédent
        combined = tf.concat([x_t, h_prev], axis=-1)   # (batch, n_feat + n_qubits)

        # Projection classique → mise à l'échelle pour AngleEmbedding
        # sigmoid × π : garantit [0, π]       (gates i, f, o)
        # tanh   × π/2: garantit [-π/2, π/2]  (gate g — plage symétrique)
        a_i = tf.math.sigmoid(self.proj_i(combined)) * np.pi
        a_f = tf.math.sigmoid(self.proj_f(combined)) * np.pi
        a_g = tf.math.tanh(self.proj_g(combined))    * (np.pi / 2.0)
        a_o = tf.math.sigmoid(self.proj_o(combined)) * np.pi

        # Évaluation VQC → expectation values PauliZ ∈ [-1, 1]
        i_q = self.qkl_i(a_i)
        f_q = self.qkl_f(a_f)
        g_q = self.qkl_g(a_g)
        o_q = self.qkl_o(a_o)

        # Activations sur les sorties quantiques → plages LSTM standards
        i_t = tf.math.sigmoid(i_q * self.gate_scale)   # input gate  [0, 1]
        f_t = tf.math.sigmoid(f_q * self.gate_scale)   # forget gate [0, 1]
        g_t = tf.math.tanh(g_q)                          # cell gate   [-1, 1] (inchange)
        o_t = tf.math.sigmoid(o_q * self.gate_scale)   # output gate [0, 1]

        # Mise à jour cellule LSTM
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * tf.math.tanh(c_t)

        return h_t, [h_t, c_t]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "n_qubits":     self.n_qubits,
            "n_vqc_layers": self.n_vqc_layers,
            "n_features":   self.n_features,
            "gate_scale":   self.gate_scale,
        })
        return cfg


# ══════════════════════════════════════════════════════════════════════
# 5. MÉMOIRE
# ══════════════════════════════════════════════════════════════════════

def _free_memory(model=None):
    if model is not None:
        del model
    tf.keras.backend.clear_session()
    gc.collect()


# ══════════════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ══════════════════════════════════════════════════════════════════════

def plot_fold(history, fold_idx, test_part, test_sess, save_dir,
              f1_mac, bal_acc, y_true=None, y_pred=None):
    epochs_range = range(1, len(history.history["loss"]) + 1)
    best_epoch   = int(np.argmin(history.history["val_loss"])) + 1

    has_cm = (y_true is not None) and (y_pred is not None)
    ncols  = 3 if has_cm else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    fig.suptitle(
        f"QLSTM  |  Fold {fold_idx} — P{test_part} S{test_sess}"
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
# 7. DONNÉES — tf.data.Dataset
# ══════════════════════════════════════════════════════════════════════

def get_window_indices(df, window_size, step_size):
    indices        = []
    current_offset = 0
    full_X = df[SIGNAL_COLS].values.astype(np.float32)
    full_y = df[COL_LABEL].values.astype(np.int32)
    for _, group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        n = len(group)
        if n >= window_size:
            for start in range(0, n - window_size + 1, step_size):
                indices.append(current_offset + start)
        current_offset += n
    return full_X, full_y, np.array(indices, dtype=np.int32)


def create_tf_dataset(df, window_size, step_size, batch_size, num_classes):
    if len(df) < window_size:
        return None, 0
    X_raw, y_raw, idxs = get_window_indices(df, window_size, step_size)
    if len(idxs) == 0:
        return None, 0

    def fetch_window(i):
        i       = int(i)
        win_X   = X_raw[i: i + window_size]
        win_y   = y_raw[i: i + window_size]
        label   = np.argmax(np.bincount(win_y, minlength=num_classes))
        return win_X, to_categorical(label, num_classes=num_classes)

    dataset = tf.data.Dataset.from_tensor_slices(idxs)
    dataset = dataset.map(
        lambda i: tf.py_function(fetch_window, [i], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, (window_size, len(SIGNAL_COLS))),
            tf.ensure_shape(y, (num_classes,)),
        )
    )
    dataset = dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    steps   = (len(idxs) + batch_size - 1) // batch_size
    return dataset, steps


def create_predict_dataset(df, window_size, step_size, batch_size, num_classes):
    """Dataset sans .repeat() : chaque fenêtre est produite exactement une fois."""
    if len(df) < window_size:
        return None, 0
    X_raw, y_raw, idxs = get_window_indices(df, window_size, step_size)
    if len(idxs) == 0:
        return None, 0

    def fetch_window(i):
        i     = int(i)
        win_X = X_raw[i: i + window_size]
        win_y = y_raw[i: i + window_size]
        label = np.argmax(np.bincount(win_y, minlength=num_classes))
        return win_X, to_categorical(label, num_classes=num_classes)

    dataset = tf.data.Dataset.from_tensor_slices(idxs)
    dataset = dataset.map(
        lambda i: tf.py_function(fetch_window, [i], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, (window_size, len(SIGNAL_COLS))),
            tf.ensure_shape(y, (num_classes,)),
        )
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    steps   = (len(idxs) + batch_size - 1) // batch_size
    return dataset, steps


def get_window_labels(df, window_size, step_size):
    y_all = []
    for _, group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        y = group[COL_LABEL].values.astype(np.int32)
        for start in range(0, len(y) - window_size + 1, step_size):
            y_all.append(np.argmax(np.bincount(y[start: start + window_size])))
    return np.array(y_all)


# ══════════════════════════════════════════════════════════════════════
# 8. CONSTRUCTION DU MODÈLE
# ══════════════════════════════════════════════════════════════════════

def build_model(input_shape, num_classes, params):
    """
    Architecture complète :

        Input (window_size, n_features)
          │
         [QLSTMCell] × n_qlstm_layers   ← VQCs remplacent les gates LSTM
          │
         Dense(dense_units, relu)        ← tête classique
          │
         Dropout
          │
         Dense(3, softmax)

    Paramètres Optuna :
      n_qubits       : qubits par VQC (expressivité vs vitesse)
      n_vqc_layers   : profondeur StronglyEntanglingLayers
      n_qlstm_layers : cellules QLSTM empilées (1 ou 2)
      dense_units    : neurones de la tête classique
    """
    n_qubits       = params["n_qubits"]
    n_vqc_layers   = params["n_vqc_layers"]
    n_qlstm_layers = params.get("n_qlstm_layers", 1)
    dense_units    = params.get("dense_units", 64)
    dropout_rate   = params.get("dropout_rate", 0.3)
    l2_strength    = params.get("l2_reg", 0.0)
    lr             = params["learning_rate"]
    gate_scale     = params.get("gate_scale", 1.0)
    reg            = l2(l2_strength) if l2_strength > 0 else None

    window_size, n_features = input_shape
    inp = Input(shape=input_shape)
    x   = inp

    for layer_idx in range(n_qlstm_layers):
        # La 2ème cellule reçoit h_t (n_qubits) de la 1ère comme entrée
        n_in       = n_features if layer_idx == 0 else n_qubits
        return_seq = (layer_idx < n_qlstm_layers - 1)
        cell = QLSTMCell(n_qubits, n_vqc_layers, n_in, gate_scale=gate_scale, name=f"qlstm_cell_{layer_idx}")
        x = tf.keras.layers.RNN(cell, return_sequences=return_seq,
                                name=f"qlstm_{layer_idx}")(x)

    x   = Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x   = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    # run_eagerly=True indispensable : PennyLane's qnode est incompatible avec
    # le tracing AutoGraph de TF (utilisé par RNN en mode graphe). En mode
    # eager, chaque timestep appelle le circuit Python directement.
    model.compile(
        optimizer=Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# 9. OPTUNA
# ══════════════════════════════════════════════════════════════════════

def optuna_objective(trial, df_splits, num_classes):
    n_qubits       = trial.suggest_categorical("n_qubits",       QLSTM_OPTUNA_SPACE["n_qubits"])
    n_vqc_layers   = trial.suggest_categorical("n_vqc_layers",   QLSTM_OPTUNA_SPACE["n_vqc_layers"])
    n_qlstm_layers = trial.suggest_categorical("n_qlstm_layers", QLSTM_OPTUNA_SPACE["n_qlstm_layers"])
    dense_units    = trial.suggest_categorical("dense_units",    QLSTM_OPTUNA_SPACE["dense_units"])
    dropout_rate   = trial.suggest_float("dropout_rate", **QLSTM_OPTUNA_SPACE["dropout_rate"])
    l2_reg         = trial.suggest_float("l2_reg",        **QLSTM_OPTUNA_SPACE["l2_reg"])
    learning_rate  = trial.suggest_float("learning_rate", **QLSTM_OPTUNA_SPACE["learning_rate"])
    batch_size     = trial.suggest_categorical("batch_size", QLSTM_OPTUNA_SPACE["batch_size"])

    params = {
        "n_qubits":       n_qubits,     "n_vqc_layers":   n_vqc_layers,
        "n_qlstm_layers": n_qlstm_layers, "dense_units":  dense_units,
        "dropout_rate":   dropout_rate, "l2_reg":         l2_reg,
        "learning_rate":  learning_rate,"batch_size":     batch_size,
    }
    scores = []

    for (df_fit, df_val) in df_splits:
        ds_fit, st_fit = create_tf_dataset(df_fit, QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE, batch_size, num_classes)
        ds_val, st_val = create_tf_dataset(df_val, QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE, batch_size, num_classes)
        if ds_fit is None or ds_val is None:
            scores.append(0.0)
            continue
        model   = None
        ds_pred = None
        try:
            model = build_model((QLSTM_WINDOW_SIZE, len(SIGNAL_COLS)), num_classes, params)
            model.fit(
                ds_fit, validation_data=ds_val,
                epochs=QLSTM_OPTUNA_EPOCHS,
                steps_per_epoch=st_fit, validation_steps=st_val,
                shuffle=False,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                    TerminateOnNaN(),
                ],
                verbose=0,
            )
            ds_pred, _ = create_predict_dataset(
                df_val, QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE, batch_size, num_classes
            )
            y_pred = np.argmax(model.predict(ds_pred, verbose=0), axis=1)
            y_true = get_window_labels(df_val, QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE)
            scores.append(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            )
        except tf.errors.ResourceExhaustedError:
            raise optuna.exceptions.TrialPruned("OOM")
        except Exception as exc:
            print(f"  [WARN] split échoué ({type(exc).__name__}: {exc})")
            scores.append(0.0)
        finally:
            _free_memory(model)
            if "ds_fit" in locals(): del ds_fit
            if "ds_val" in locals(): del ds_val
            if ds_pred is not None: del ds_pred
            gc.collect()

    return float(np.mean(scores)) if scores else 0.0


def optimize_hyperparams(df, num_classes):
    n_trials = QLSTM_OPTUNA_TRIALS
    n_sess   = QLSTM_OPTUNA_SESSIONS
    print(f"\nOPTUNA {MODEL_NAME} — {n_trials} trials | {n_sess} sessions\n" + "=" * 60)
    import random
    random.seed(42)  # reproductibilité des sessions de validation Optuna

    unique_sessions = [tuple(x) for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
    val_sessions    = random.sample(unique_sessions, min(n_sess, len(unique_sessions)))
    print(f"Sessions Optuna : {val_sessions}")

    df_splits = []
    for (val_part, val_sess) in val_sessions:
        df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
        df_val   = df[ (df[COL_PARTICIPANT] == val_part)  & (df[COL_SESSION] == val_sess)].copy()

        # SMOTE seulement sur df_train, avant le scaling
        df_train = resample_dataframe(df_train, SIGNAL_COLS)

        scaler   = RobustScaler()
        df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
        df_val[SIGNAL_COLS]   = scaler.transform(df_val[SIGNAL_COLS])
        df_splits.append((df_train, df_val))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, df_splits, num_classes),
        n_trials=n_trials, show_progress_bar=True,
        n_jobs=1, gc_after_trial=True, catch=(Exception,),
    )
    del df_splits; gc.collect()

    print(f"\nBest F1-Macro : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    OPTUNA_PATH_QLSTM.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_QLSTM, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return study.best_params, val_sessions


# ══════════════════════════════════════════════════════════════════════
# 10. MODÈLE GLOBAL
# ══════════════════════════════════════════════════════════════════════

def train_global_model(df, best_params, num_classes, val_sessions):
    print("\n" + "=" * 60 + f"\nMODÈLE GLOBAL — {MODEL_NAME}\n" + "=" * 60)

    val_sessions_set = set(val_sessions)
    mask_val  = df.set_index([COL_PARTICIPANT, COL_SESSION]).index.isin(val_sessions_set)
    df_train  = df[~mask_val].copy()
    df_val    = df[mask_val].copy()

    # SMOTE seulement sur df_train, avant le scaling
    df_train = resample_dataframe(df_train, SIGNAL_COLS)

    scaler = RobustScaler()
    df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
    df_val[SIGNAL_COLS]   = scaler.transform(df_val[SIGNAL_COLS])

    y_train_labels = get_window_labels(df_train, QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE)
    w = compute_class_weight("balanced", classes=np.unique(y_train_labels), y=y_train_labels)

    bs = best_params.get("batch_size", 16)
    ds_train, st_train = create_tf_dataset(df_train, QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE, bs, num_classes)
    ds_val,   st_val   = create_tf_dataset(df_val,   QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE, bs, num_classes)

    model = None
    try:
        model = build_model((QLSTM_WINDOW_SIZE, len(SIGNAL_COLS)), num_classes, best_params)
        monitor   = "val_loss" if ds_val is not None else "loss"
        callbacks = [
            EarlyStopping(monitor=monitor, patience=10, restore_best_weights=True),
            TerminateOnNaN(),
        ]
        model.fit(
            ds_train, epochs=LOSO_EPOCHS,
            validation_data=ds_val,
            steps_per_epoch=st_train, validation_steps=st_val,
            shuffle=False, callbacks=callbacks,
            class_weight=dict(zip(np.unique(y_train_labels), w)), verbose=1,
        )

        models_dir = MODELS_DIR / "QLSTM"
        models_dir.mkdir(parents=True, exist_ok=True)
        keras_path = models_dir / f"{MODEL_NAME}_global.keras"
        model.save(keras_path)
        print(f"  Modele natif sauvegarde : {keras_path}")

        if ds_val is not None:
            ds_val_pred, _ = create_predict_dataset(
                df_val, QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE, bs, num_classes
            )
            y_pred_val = np.argmax(model.predict(ds_val_pred, verbose=0), axis=1)
            y_true_val = get_window_labels(df_val, QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE)
            edge_metrics = {
                "native_qlstm": {
                    "f1_macro": float(f1_score(y_true_val, y_pred_val, average="macro", zero_division=0)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true_val, y_pred_val)),
                    "model_size_kb": round(keras_path.stat().st_size / 1024, 2),
                    "deployment_status": (
                        "export_stm32_non_tente : cellule VQC (PennyLane lightning.qubit, "
                        "boucle Python native par timestep via tf.map_fn en mode eager) "
                        "structurellement differente de fixed_layer_matrix (QRC) -- pas de "
                        "simplification equivalente identifiee a ce stade. A investiguer "
                        "separement si un export est souhaite, pas suppose impossible."
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
            del ds_val_pred
        else:
            print("  Pas de ds_val disponible, edge_comparison non calculee.")
    except Exception as exc:
        print(f"  Modèle global échoué ({type(exc).__name__}: {exc})")
    finally:
        _free_memory(model)
        if "ds_train" in locals(): del ds_train
        if "ds_val"   in locals(): del ds_val
        del df_train, df_val
        gc.collect()


# ══════════════════════════════════════════════════════════════════════
# 11. BOUCLE LOSO
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

    if USE_OPTUNA_QLSTM:
        best_params, val_sessions = optimize_hyperparams(df, num_classes)
    else:
        best_params = MODEL_PARAMS["QLSTM"]
        import random as _rnd
        _rnd.seed(42)
        all_sess    = [tuple(x) for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
        val_sessions = _rnd.sample(all_sess, min(QLSTM_OPTUNA_SESSIONS, len(all_sess)))

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

        bs = best_params.get("batch_size", 16)
        ds_fit,  st_fit  = create_tf_dataset(df_fit,  QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE, bs, num_classes)
        ds_val,  st_val  = create_tf_dataset(df_val,  QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE, bs, num_classes)
        ds_test, st_test = create_predict_dataset(df_test, QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE, bs, num_classes)

        if ds_test is None:
            print(f"  WARNING: test vide pour P{test_part}. Skip."); continue
        if ds_val is None or ds_fit is None:
            print(f"  WARNING: Fit/Val vide. Skip."); continue

        y_fitLabels = get_window_labels(df_fit, QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE)
        if len(y_fitLabels) == 0:
            continue
        w = compute_class_weight("balanced", classes=np.unique(y_fitLabels), y=y_fitLabels)

        model = None
        try:
            model = build_model((QLSTM_WINDOW_SIZE, len(SIGNAL_COLS)), num_classes, best_params)
            history = model.fit(
                ds_fit, epochs=LOSO_EPOCHS,
                validation_data=ds_val,
                steps_per_epoch=st_fit, validation_steps=st_val,
                shuffle=False,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                    TerminateOnNaN(),
                ],
                class_weight=dict(zip(np.unique(y_fitLabels), w)),
                verbose=1,
            )

            y_pred = np.argmax(model.predict(ds_test, verbose=0), axis=1)
            y_true = get_window_labels(df_test, QLSTM_WINDOW_SIZE, QLSTM_STEP_SIZE)

            f1_mac  = f1_score(y_true, y_pred, average="macro")
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            print(f"  F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            plots_dir = BASE_DIR / "training_curves" / "QLSTM"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_fold(history, test_idx + 1, test_part, test_sess,
                      plots_dir, f1_mac, bal_acc, y_true, y_pred)

            all_metrics.append({
                "Fold": test_idx + 1, "Participant": int(test_part), "Session": int(test_sess),
                "F1_Macro": float(f1_mac), "Balanced_Accuracy": float(bal_acc),
            })

        except Exception as exc:
            print(f"  Fold {test_idx+1} échoué ({type(exc).__name__}: {exc})")
        finally:
            _free_memory(model)
            del ds_fit, ds_val, ds_test
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
