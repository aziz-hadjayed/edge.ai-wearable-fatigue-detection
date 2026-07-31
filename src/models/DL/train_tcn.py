import ctypes
import gc
import json
import os
import random
import sys
import traceback
import warnings
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'

# ══════════════════════════════════════════════════════════════════════
# 1. IMPORTS
# ══════════════════════════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Supprimer la verbosité lors de la conversion TFLite
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow.lite')

tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import *

from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Conv1D,
    Dense, Dropout, GlobalAveragePooling1D, Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from models.semi_supervised import add_pseudo_labels, extract_all_windows
from utils.apply_smote import resample_dataframe

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss pour multi-classe.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fixed

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ══════════════════════════════════════════════════════════════════════
# 2. CONSTANTES
# ══════════════════════════════════════════════════════════════════════
MODEL_NAME        = "TCN"
LABEL_MAPPING     = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES      = ["baseline", "activity", "pre_fatigue", "fatigue"]
N_OPTUNA_SESSIONS  = 15      # réduit : moins de folds par trial → moins de RAM cumulée
N_OPTUNA_TRIALS    = 50     # réduit : budget serré sur 16 GB RAM
N_OPTUNA_EPOCHS    = 5      # réduit : convergence rapide suffit pour le ranking
MAX_OPTUNA_SAMPLES = 2000   # plafond fenêtres train par fold Optuna (évite OOM)

# Budget Flash STM32 pour TFLite INT8
EDGE_FLASH_BUDGET_KB = STM32_FLASH_KB

# ══════════════════════════════════════════════════════════════════════
# 3. MÉMOIRE
# ══════════════════════════════════════════════════════════════════════
def _free_memory(model=None):
    if model is not None:
        del model
    tf.keras.backend.clear_session()
    gc.collect()
    try:
        # Rend explicitement la mémoire libre au kernel Linux (évite fragmentation)
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════
# 4. EXPORT STM32 — TFLite INT8 + C header
# ══════════════════════════════════════════════════════════════════════
def _export_c_array(tflite_bytes, h_path, stem):
    var_name = stem.replace("-", "_").replace(".", "_").lower()
    guard    = var_name.upper() + "_H"
    hex_vals = [f"0x{b:02x}" for b in tflite_bytes]
    rows     = ["  " + ", ".join(hex_vals[i:i+16]) for i in range(0, len(hex_vals), 16)]
    lines = [
        f"/* Auto-generated for STM32H7A3ZIT6Q — X-CUBE-AI */",
        f"#ifndef {guard}", f"#define {guard}", "",
        f"const unsigned char {var_name}[] = {{",
        *[r + "," for r in rows], f"}};",
        f"const unsigned int {var_name}_len = {len(tflite_bytes)};",
        "", f"#endif /* {guard} */",
    ]
    h_path.write_text("\n".join(lines))

def _silent_tflite_convert(converter):
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        return converter.convert()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)

def _save_tflite_int8(model, X_representative, models_dir, stem):
    def representative_dataset():
        idx = np.random.default_rng(42).choice(
            len(X_representative), min(200, len(X_representative)), replace=False
        )
        for i in idx:
            yield [X_representative[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations              = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset     = representative_dataset
    converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type       = tf.int8
    converter.inference_output_type      = tf.int8
    tflite_bytes = _silent_tflite_convert(converter)

    tflite_path = models_dir / f"{stem}_int8.tflite"
    tflite_path.write_bytes(tflite_bytes)
    _export_c_array(tflite_bytes, models_dir / f"{stem}_int8.h", stem)
    print(f"  TFLite : {tflite_path.name}  ({len(tflite_bytes)/1024:.1f} KB / {EDGE_FLASH_BUDGET_KB} KB)")

# ══════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ══════════════════════════════════════════════════════════════════════
def plot_fold(history, fold_idx, test_part, test_sess, save_dir,
              f1_mac, bal_acc, y_true=None, y_pred=None):
    import seaborn as sns
    epochs_range = range(1, len(history.history["loss"]) + 1)
    best_epoch   = int(np.argmin(history.history["val_loss"])) + 1

    has_cm = (y_true is not None) and (y_pred is not None)
    ncols  = 3 if has_cm else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    fig.suptitle(
        f"TCN  |  Fold {fold_idx} — P{test_part} S{test_sess}"
        f"  |  F1-Macro: {f1_mac:.3f}  |  Bal.Acc: {bal_acc:.3f}",
        fontsize=13, fontweight="bold"
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
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
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
# 6. DONNÉES — fenêtrage
# ══════════════════════════════════════════════════════════════════════
def get_window_indices(df, window_size, step_size, num_classes):
    indices = []
    groups = df.groupby([COL_PARTICIPANT, COL_SESSION])
    
    full_X = df[SIGNAL_COLS].values.astype(np.float32)
    full_y = df[COL_LABEL].values.astype(np.int32)
    current_offset = 0
    
    for _, group in groups:
        n = len(group)
        if n >= window_size:
            for start in range(0, n - window_size + 1, step_size):
                indices.append(current_offset + start)
        current_offset += n
    return full_X, full_y, np.array(indices, dtype=np.int32)

def create_tf_dataset(df, window_size, step_size, batch_size, num_classes):
    if len(df) < window_size:
        return None, 0
    
    X_raw, y_raw, idxs = get_window_indices(df, window_size, step_size, num_classes)
    if len(idxs) == 0:
        return None, 0

    def fetch_window(i):
        i = int(i)
        win_X = X_raw[i : i + window_size]
        win_y = y_raw[i : i + window_size]
        counts = np.bincount(win_y, minlength=num_classes)
        label = np.argmax(counts)
        return win_X, to_categorical(label, num_classes=num_classes)

    dataset = tf.data.Dataset.from_tensor_slices(idxs)
    dataset = dataset.map(
        lambda i: tf.py_function(fetch_window, [i], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(lambda x, y: (tf.ensure_shape(x, (window_size, len(SIGNAL_COLS))), 
                                       tf.ensure_shape(y, (num_classes,))))
    
    dataset = dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    steps = (len(idxs) + batch_size - 1) // batch_size
    return dataset, steps

def get_window_labels(df, window_size, step_size):
    y_all = []
    groups = df.groupby([COL_PARTICIPANT, COL_SESSION])
    for _, group in groups:
        y = group[COL_LABEL].values.astype(np.int32)
        for start in range(0, len(y) - window_size + 1, step_size):
            counts = np.bincount(y[start : start + window_size])
            y_all.append(np.argmax(counts))
    return np.array(y_all)


# ══════════════════════════════════════════════════════════════════════
# 7. MODÈLE TCN PUR + OPTUNA
# ══════════════════════════════════════════════════════════════════════
def _tcn_residual_block(x, filters, kernel_size, dilation, dropout_rate, activation_name, l2_reg):
    """
    Bloc résiduel TCN :
      2× Conv1D causale dilatée → BN → Activation → Dropout
      + connexion résiduelle (skip connection)
    """
    shortcut = x
    h = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation,
               use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    h = BatchNormalization()(h)
    h = Activation(activation_name)(h)
    h = Dropout(dropout_rate)(h)
    h = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation,
               use_bias=False, kernel_regularizer=l2(l2_reg))(h)
    h = BatchNormalization()(h)
    h = Activation(activation_name)(h)
    h = Dropout(dropout_rate)(h)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding="same", use_bias=False,
                          kernel_regularizer=l2(l2_reg))(shortcut)
    return Activation(activation_name)(Add()([shortcut, h]))

def build_model(trial, input_shape, num_classes):
    """
    Architecture TCN pure :
      Conv1D initiale → BN → Act  (projection vers tcn_filters)
      N blocs TCN résiduels dilatés (dilation=1,2,4,8…)
      Tête : GAP → Dense → Dropout → Softmax(3)
    """
    n_tcn_blocks  = trial.suggest_categorical("n_tcn_blocks", TCN_OPTUNA_SPACE["n_tcn_blocks"])
    tcn_filters   = trial.suggest_categorical("tcn_filters",  TCN_OPTUNA_SPACE["tcn_filters"])
    tcn_kernel    = trial.suggest_categorical("tcn_kernel",   TCN_OPTUNA_SPACE["tcn_kernel"])
    activation    = trial.suggest_categorical("activation",   TCN_OPTUNA_SPACE["activation"])
    l2_reg        = trial.suggest_float("l2_reg",       *TCN_OPTUNA_SPACE["l2_reg"], log=True)
    optimizer_str = trial.suggest_categorical("optimizer",    TCN_OPTUNA_SPACE["optimizer"])
    dense_units   = trial.suggest_categorical("dense_units",  TCN_OPTUNA_SPACE["dense_units"])
    dropout_rate  = trial.suggest_float("dropout_rate",  *TCN_OPTUNA_SPACE["dropout_rate"])
    learning_rate = trial.suggest_float("learning_rate", *TCN_OPTUNA_SPACE["learning_rate"], log=True)

    # --- Entrée + projection initiale ---
    inputs = Input(shape=input_shape)
    x = Conv1D(tcn_filters, 1, padding="causal", use_bias=False,
               kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    # --- Pile de blocs TCN dilatés ---
    for i in range(n_tcn_blocks):
        x = _tcn_residual_block(x, tcn_filters, tcn_kernel,
                                dilation=2**i, dropout_rate=dropout_rate,
                                activation_name=activation, l2_reg=l2_reg)

    # --- Tête de classification ---
    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation=activation, kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="TCN")
    opt = Adam(learning_rate) if optimizer_str == "adam" else RMSprop(learning_rate)
    
    # Utilisation de Focal Loss comme demandé
    model.compile(optimizer=opt, 
                  loss="categorical_crossentropy",
                  metrics=["accuracy"], jit_compile=False)
    return model

def optuna_objective(trial, df, num_classes, n_optuna_epochs=N_OPTUNA_EPOCHS):
    config      = WINDOW_CONFIGS["default"]
    window_size = config["window_size"]
    step_size   = config["step_size"]
    input_shape = (window_size, len(SIGNAL_COLS))
    batch_size  = trial.suggest_categorical("batch_size", TCN_OPTUNA_SPACE["batch_size"])

    all_sessions  = list(df.groupby([COL_PARTICIPANT, COL_SESSION]).groups.keys())
    eval_sessions = random.sample(all_sessions, min(N_OPTUNA_SESSIONS, len(all_sessions)))
    scores = []

    for fold_idx, (val_part, val_sess) in enumerate(eval_sessions):
        df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
        df_val   = df[ (df[COL_PARTICIPANT] == val_part)  & (df[COL_SESSION] == val_sess)].copy()

        # SMOTE seulement sur df_train, avant le scaling et le windowing
        df_train = resample_dataframe(df_train, SIGNAL_COLS)

        scaler = RobustScaler()
        df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
        df_val[SIGNAL_COLS]   = scaler.transform(df_val[SIGNAL_COLS])

        ds_train, st_train = create_tf_dataset(df_train, window_size, step_size, batch_size, num_classes)
        ds_val,   st_val   = create_tf_dataset(df_val,   window_size, step_size, batch_size, num_classes)

        if ds_train is None or ds_val is None:
            scores.append(0.0); continue

        w = compute_class_weight("balanced", classes=np.unique(df_train[COL_LABEL]), y=df_train[COL_LABEL])
        class_weight_dict = dict(zip(np.unique(df_train[COL_LABEL]), w))

        model = None
        try:
            _free_memory()
            model = build_model(trial, input_shape, num_classes)

            model_kb = model.count_params() / 1024
            if model_kb > EDGE_FLASH_BUDGET_KB:
                raise optuna.exceptions.TrialPruned(f"Flash {model_kb:.0f} KB > {EDGE_FLASH_BUDGET_KB} KB")

            model.fit(
                ds_train,
                epochs=n_optuna_epochs, 
                steps_per_epoch=st_train,
                validation_data=ds_val,
                validation_steps=st_val,
                callbacks=[TerminateOnNaN()], 
                class_weight=class_weight_dict,
                verbose=0,
            )
            y_pred = np.argmax(model.predict(ds_val, steps=st_val, verbose=0), axis=1)
            y_val_true = get_window_labels(df_val, window_size, step_size)
            
            min_len = min(len(y_pred), len(y_val_true))
            score  = f1_score(y_val_true[:min_len], y_pred[:min_len], average="macro", zero_division=0)
            scores.append(score)

            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            print(f"  [WARN] fold échoué ({type(exc).__name__}: {exc})")
            scores.append(0.0)
        finally:
            _free_memory(model)
            if 'ds_train' in locals(): del ds_train
            if 'ds_val'   in locals(): del ds_val
            gc.collect()

    return float(np.mean(scores)) if scores else 0.0

def optimize_hyperparams(df, num_classes, n_trials=N_OPTUNA_TRIALS, n_optuna_epochs=N_OPTUNA_EPOCHS):
    print(f"\nOPTUNA TCN — {n_trials} trials | {N_OPTUNA_SESSIONS} sessions\n" + "=" * 60)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    # df passé directement — fenêtres extraites à la demande dans l'objectif
    study.optimize(
        lambda trial: optuna_objective(trial, df, num_classes, n_optuna_epochs),
        n_trials=n_trials, show_progress_bar=True,
        gc_after_trial=True, catch=(Exception,),
    )
    gc.collect()

    print(f"\nBest F1-Macro : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    OPTUNA_PATH_TCN.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_TCN, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return study.best_params

# ══════════════════════════════════════════════════════════════════════
# 9. MODÈLE GLOBAL
# ══════════════════════════════════════════════════════════════════════
def _compute_classification_metrics(y_true, y_pred):
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_fatigue": float(precision_score(y_true, y_pred, labels=[3], average="macro", zero_division=0)),
        "recall_fatigue": float(recall_score(y_true, y_pred, labels=[3], average="macro", zero_division=0)),
    }


def _predict_tflite(tflite_path, X_data):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details["index"]
    output_index = output_details["index"]
    input_scale, input_zero_point = input_details.get("quantization", (0.0, 0))
    output_scale, output_zero_point = output_details.get("quantization", (0.0, 0))
    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]
    preds = []

    for sample in X_data:
        x = sample[np.newaxis, ...].astype(np.float32)
        if np.issubdtype(input_dtype, np.integer):
            x = np.round(x / input_scale + input_zero_point)
            x = np.clip(x, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
        else:
            x = x.astype(input_dtype)

        interpreter.set_tensor(input_index, x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_index)
        if np.issubdtype(output_dtype, np.integer):
            y = (y.astype(np.float32) - output_zero_point) * output_scale
        preds.append(int(np.argmax(y, axis=1)[0]))

    return np.array(preds)


def _save_edge_comparison_metrics(model_name, edge_metrics):
    curr_metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                content = f.read().strip()
            curr_metrics = json.loads(content) if content else {}
        except (json.JSONDecodeError, ValueError):
            print(f"⚠ metrics.json invalide ou vide, réinitialisation : {METRICS_PATH}")
            curr_metrics = {}

    curr_metrics.setdefault(model_name, {})
    curr_metrics[model_name]["edge_comparison"] = edge_metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(curr_metrics, f, indent=4)


def _apply_pruning_and_finetune(model, X_tr, y_tr, X_vl, y_vl, weight_dict,
                                final_sparsity=0.5, pruning_epochs=20, batch_size=32):
    num_classes = len(LABEL_MAPPING)
    steps_per_epoch = max(1, len(X_tr) // batch_size)
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=final_sparsity,
            begin_step=0,
            end_step=steps_per_epoch * pruning_epochs,
        )
    }
    model_prunable = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    learning_rate = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    if isinstance(model.optimizer, RMSprop):
        optimizer = RMSprop(learning_rate)
    else:
        optimizer = Adam(learning_rate)

    model_prunable.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model_prunable.fit(
        X_tr, to_categorical(y_tr, num_classes),
        validation_data=(X_vl, to_categorical(y_vl, num_classes)),
        epochs=pruning_epochs,
        batch_size=batch_size,
        class_weight=weight_dict,
        callbacks=[
            tfmot.sparsity.keras.UpdatePruningStep(),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            TerminateOnNaN(),
        ],
        verbose=1,
    )
    return tfmot.sparsity.keras.strip_pruning(model_prunable)


def train_global_model(df_labeled, df_unlabeled, best_params, num_classes):
    print("\n" + "=" * 60 + f"\nMODÈLE GLOBAL — {MODEL_NAME}\n" + "=" * 60)
    config = WINDOW_CONFIGS["default"]
    W_SIZE, S_SIZE, EPOCHS = config["window_size"], config["step_size"], config["epochs"]
    batch_size = best_params.get("batch_size", 32)

    df_all = df_labeled.copy()
    scaler = RobustScaler()
    df_all[SIGNAL_COLS] = scaler.fit_transform(df_all[SIGNAL_COLS])
    df_unl = df_unlabeled.copy()
    if len(df_unl) > 0:
        df_unl[SIGNAL_COLS] = scaler.transform(df_unl[SIGNAL_COLS])

    X_all, y_all = extract_all_windows(df_all, W_SIZE, S_SIZE)
    X_unlabeled, _ = (
        extract_all_windows(df_unl, W_SIZE, S_SIZE)
        if len(df_unl) > 0 else (np.array([]), np.array([]))
    )
    del df_all, df_unl
    gc.collect()
    print(f"  Fenêtres labellisées : {len(X_all)} | unlabeled : {len(X_unlabeled)}")

    model = None
    try:
        idx = np.random.default_rng(42).permutation(len(X_all))
        split = int(0.9 * len(X_all))
        X_tr, y_tr = X_all[idx[:split]], y_all[idx[:split]]
        X_vl, y_vl = X_all[idx[split:]], y_all[idx[split:]]

        w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        weight_dict = dict(zip(np.unique(y_tr), w))

        model = build_model(
            optuna.trial.FixedTrial(best_params),
            (W_SIZE, len(SIGNAL_COLS)), num_classes
        )
        model.fit(
            X_tr, to_categorical(y_tr, num_classes),
            validation_data=(X_vl, to_categorical(y_vl, num_classes)),
            epochs=EPOCHS,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                       TerminateOnNaN()],
            class_weight=weight_dict,
            verbose=1,
        )

        X_tr_v2, y_tr_v2, pseudo_y, _ = add_pseudo_labels(model, X_tr, y_tr, X_unlabeled)
        if len(pseudo_y) > 0:
            _free_memory(model)
            model = build_model(
                optuna.trial.FixedTrial(best_params),
                (W_SIZE, len(SIGNAL_COLS)), num_classes
            )
            w_v2 = compute_class_weight("balanced", classes=np.unique(y_tr_v2), y=y_tr_v2)
            weight_dict_v2 = dict(zip(np.unique(y_tr_v2), w_v2))
            model.fit(
                X_tr_v2, to_categorical(y_tr_v2, num_classes),
                validation_data=(X_vl, to_categorical(y_vl, num_classes)),
                epochs=EPOCHS,
                batch_size=batch_size,
                callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                           TerminateOnNaN()],
                class_weight=weight_dict_v2,
                verbose=1,
            )
            X_repr = np.concatenate([X_tr_v2, X_vl], axis=0)
        else:
            X_repr = X_all

        models_dir = MODELS_DIR / "TCN"
        models_dir.mkdir(parents=True, exist_ok=True)
        edge_metrics = {}

        keras_path = models_dir / f"{MODEL_NAME}_global_float32.keras"
        model.save(keras_path)
        y_pred_f32 = np.argmax(model.predict(X_vl, verbose=0), axis=1)
        edge_metrics["float32"] = _compute_classification_metrics(y_vl, y_pred_f32)
        edge_metrics["float32"]["model_size_kb"] = round(keras_path.stat().st_size / 1024, 2)

        _save_tflite_int8(model, X_repr, models_dir, f"{MODEL_NAME}_global")
        tflite_path = models_dir / f"{MODEL_NAME}_global_int8.tflite"
        y_pred_int8 = _predict_tflite(tflite_path, X_vl)
        edge_metrics["int8_tflite"] = _compute_classification_metrics(y_vl, y_pred_int8)
        edge_metrics["int8_tflite"]["model_size_kb"] = round(tflite_path.stat().st_size / 1024, 2)

        model_pruned = None
        try:
            model_pruned = _apply_pruning_and_finetune(model, X_tr, y_tr, X_vl, y_vl, weight_dict)
            keras_pruned_path = models_dir / f"{MODEL_NAME}_global_pruned_float32.keras"
            model_pruned.save(keras_pruned_path)
            y_pred_pruned_f32 = np.argmax(model_pruned.predict(X_vl, verbose=0), axis=1)
            edge_metrics["pruned_float32"] = _compute_classification_metrics(y_vl, y_pred_pruned_f32)
            edge_metrics["pruned_float32"]["model_size_kb"] = round(keras_pruned_path.stat().st_size / 1024, 2)

            _save_tflite_int8(model_pruned, X_repr, models_dir, f"{MODEL_NAME}_global_pruned")
            tflite_pruned_path = models_dir / f"{MODEL_NAME}_global_pruned_int8.tflite"
            y_pred_pruned_int8 = _predict_tflite(tflite_pruned_path, X_vl)
            edge_metrics["pruned_int8_tflite"] = _compute_classification_metrics(y_vl, y_pred_pruned_int8)
            edge_metrics["pruned_int8_tflite"]["model_size_kb"] = round(tflite_pruned_path.stat().st_size / 1024, 2)
        except Exception as exc:
            print(f"  ❌ Pruning échoué ({type(exc).__name__}: {exc})")
            traceback.print_exc()
        finally:
            _free_memory(model_pruned)

        _save_edge_comparison_metrics(MODEL_NAME, edge_metrics)
        print(f"  ✅ Modèle global exporté (.tflite + .h).")
    except Exception as exc:
        print(f"  ❌ Modèle global échoué ({type(exc).__name__}: {exc})")
        traceback.print_exc()
    finally:
        _free_memory(model)
        gc.collect()

# ══════════════════════════════════════════════════════════════════════
# 8. BOUCLE LOSO
# ══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60 + f"\nTRAIN LOSO — {MODEL_NAME}\n" + "=" * 60)
    if not DATA_MODEL_READY.exists():
        return print(f"❌ Dataset non trouvé : {DATA_MODEL_READY}")

    df = pd.read_csv(DATA_MODEL_READY)
    df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors="coerce").fillna(-1).astype(int)
    df_labeled = df[df[COL_LABEL] >= 0].copy()
    df_unlabeled = df[df[COL_LABEL] == -1].copy()
    print(f"Labeled: {len(df_labeled)} lignes | Unlabeled: {len(df_unlabeled)} lignes")

    unique_sessions = [
        tuple(x) for x in
        df_labeled[df_labeled[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
    ]
    num_classes = len(LABEL_MAPPING)
    print(f"Sessions détectées : {len(unique_sessions)}")

    best_params = (
        optimize_hyperparams(df_labeled, num_classes)
        if USE_OPTUNA_TCN
        else MODEL_PARAMS["TCN"]
    )

    all_metrics = []
    random.seed(42)

    for test_idx, (test_part, test_sess) in enumerate(unique_sessions):
        print(f"\n--- FOLD {test_idx+1}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")
        config = WINDOW_CONFIGS.get(test_part, WINDOW_CONFIGS["default"])
        W_SIZE, S_SIZE, EPOCHS = config["window_size"], config["step_size"], config["epochs"]

        df_pool = df_labeled[~((df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess))].copy()
        df_test = df_labeled[ (df_labeled[COL_PARTICIPANT] == test_part)  & (df_labeled[COL_SESSION] == test_sess)].copy()

        pool_sessions      = [tuple(x) for x in df_pool[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
        val_part, val_sess = random.choice(pool_sessions)
        print(f"  Val : P{val_part} S{val_sess}")

        df_fit = df_pool[~((df_pool[COL_PARTICIPANT] == val_part) & (df_pool[COL_SESSION] == val_sess))].copy()
        df_val = df_pool[ (df_pool[COL_PARTICIPANT] == val_part)  & (df_pool[COL_SESSION] == val_sess)].copy()

        # SMOTE seulement sur df_fit, avant le scaling et le windowing
        df_fit = resample_dataframe(df_fit, SIGNAL_COLS)

        scaler = RobustScaler()
        df_fit[SIGNAL_COLS]  = scaler.fit_transform(df_fit[SIGNAL_COLS])
        df_val[SIGNAL_COLS]  = scaler.transform(df_val[SIGNAL_COLS])
        df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])
        df_unlabeled_fit = df_unlabeled.copy()
        if len(df_unlabeled_fit) > 0:
            df_unlabeled_fit[SIGNAL_COLS] = scaler.transform(df_unlabeled_fit[SIGNAL_COLS])

        X_fit, y_fit = extract_all_windows(df_fit, W_SIZE, S_SIZE)
        X_val_np, y_val_np = extract_all_windows(df_val, W_SIZE, S_SIZE)
        X_unlabeled, _ = (
            extract_all_windows(df_unlabeled_fit, W_SIZE, S_SIZE)
            if len(df_unlabeled_fit) > 0 else (np.array([]), np.array([]))
        )

        ds_fit, st_fit = create_tf_dataset(df_fit, W_SIZE, S_SIZE, best_params.get("batch_size", 32), num_classes)
        ds_val, st_val = create_tf_dataset(df_val, W_SIZE, S_SIZE, best_params.get("batch_size", 32), num_classes)
        ds_test, st_test = create_tf_dataset(df_test, W_SIZE, S_SIZE, best_params.get("batch_size", 32), num_classes)

        if ds_fit is None or ds_val is None or ds_test is None:
            print(f"  WARNING: split vide. Skip."); continue

        w = compute_class_weight("balanced", classes=np.unique(df_fit[COL_LABEL]), y=df_fit[COL_LABEL])
        weight_dict = dict(zip(np.unique(df_fit[COL_LABEL]), w))

        model = None
        try:
            model = build_model(
                optuna.trial.FixedTrial(best_params),
                (W_SIZE, len(SIGNAL_COLS)), num_classes
            )
            history = model.fit(
                ds_fit,
                epochs=EPOCHS,
                steps_per_epoch=st_fit,
                validation_data=ds_val,
                validation_steps=st_val,
                callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                           TerminateOnNaN()],
                class_weight=weight_dict,
                verbose=1,
            )

            X_fit_v2, y_fit_v2, pseudo_y, _ = add_pseudo_labels(model, X_fit, y_fit, X_unlabeled)
            if len(pseudo_y) > 0:
                _free_memory(model)
                model = build_model(
                    optuna.trial.FixedTrial(best_params),
                    (W_SIZE, len(SIGNAL_COLS)), num_classes
                )
                w_v2 = compute_class_weight("balanced", classes=np.unique(y_fit_v2), y=y_fit_v2)
                weight_dict_v2 = dict(zip(np.unique(y_fit_v2), w_v2))
                history = model.fit(
                    X_fit_v2, to_categorical(y_fit_v2, num_classes),
                    epochs=EPOCHS,
                    validation_data=(X_val_np, to_categorical(y_val_np, num_classes)),
                    batch_size=best_params.get("batch_size", 32),
                    callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                               TerminateOnNaN()],
                    class_weight=weight_dict_v2,
                    verbose=1,
                )

            y_pred  = np.argmax(model.predict(ds_test, steps=st_test, verbose=0), axis=1)
            y_test_true = get_window_labels(df_test, W_SIZE, S_SIZE)
            
            min_len = min(len(y_pred), len(y_test_true))
            f1_mac  = f1_score(y_test_true[:min_len], y_pred[:min_len], average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test_true[:min_len], y_pred[:min_len])
            print(f"  ✅ F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            plots_dir = BASE_DIR / "training_curves" / "TCN"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_fold(history, test_idx+1, test_part, test_sess,
                      plots_dir, f1_mac, bal_acc, y_test_true[:min_len], y_pred[:min_len])

            all_metrics.append({
                "Fold": test_idx+1, "Participant": int(test_part), "Session": int(test_sess),
                "F1_Macro": float(f1_mac), "Balanced_Accuracy": float(bal_acc),
            })

        except Exception as exc:
            print(f"  ❌ Fold {test_idx+1} échoué ({type(exc).__name__}: {exc})")
        finally:
            _free_memory(model)
            if 'ds_fit' in locals(): del ds_fit
            if 'ds_val' in locals(): del ds_val
            if 'ds_test' in locals(): del ds_test
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
            print(f"⚠ metrics.json invalide ou vide, réinitialisation : {METRICS_PATH}")
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
    print(f"\n📊 Métriques → {METRICS_PATH}")

    train_global_model(df_labeled, df_unlabeled, best_params, num_classes)


if __name__ == "__main__":
    main()
