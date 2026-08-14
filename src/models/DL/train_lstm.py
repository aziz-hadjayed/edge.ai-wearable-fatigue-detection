import gc
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

# Limit CPU parallelism to avoid high CPU usage
os.environ['TF_USE_LEGACY_KERAS'] = '1'


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '0'
os.environ['TF_ADALOG_LEVEL'] = '3'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

if not tf.keras.__name__.startswith("tf_keras"):
    print(
        "⚠ Keras 3 détecté malgré TF_USE_LEGACY_KERAS=1 — "
        "installez le package 'tf_keras' (pip install tf_keras) pour que "
        "tensorflow_model_optimization fonctionne correctement."
    )
# ══════════════════════════════════════════════════════════════════════
# 1. IMPORTS
# ══════════════════════════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
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

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from models.semi_supervised import add_pseudo_labels, extract_all_windows

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ══════════════════════════════════════════════════════════════════════
# 2. CONSTANTES
# ══════════════════════════════════════════════════════════════════════
MODEL_NAME        = "LSTM"
LABEL_MAPPING     = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES      = ["baseline", "activity", "pre_fatigue", "fatigue"]
N_OPTUNA_SESSIONS = 25       # sessions tirées au hasard pour évaluer chaque trial Optuna

# ══════════════════════════════════════════════════════════════════════
# 3. MÉMOIRE
# ══════════════════════════════════════════════════════════════════════
def _free_memory(model=None):
    if model is not None:
        del model
    tf.keras.backend.clear_session()
    gc.collect()

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

def _clone_model_for_tflite(model):
    def _lstm_config_for_tflite(config):
        config = dict(config)
        # Compatibilité Keras/tf_keras : certaines versions acceptent
        # use_cudnn, d'autres lèvent "Keyword argument not understood".
        config.pop("use_cudnn", None)
        config["unroll"] = True
        config["stateful"] = False
        return config

    def clone_layer(layer):
        if isinstance(layer, LSTM):
            return LSTM.from_config(_lstm_config_for_tflite(layer.get_config()))

        if isinstance(layer, Bidirectional):
            config = layer.get_config()
            for key in ("layer", "forward_layer", "backward_layer"):
                inner = config.get(key)
                if isinstance(inner, dict) and inner.get("class_name") == "LSTM":
                    inner_config = _lstm_config_for_tflite(inner.get("config", {}))
                    config[key] = {**inner, "config": inner_config}
            return Bidirectional.from_config(config)

        return layer.__class__.from_config(layer.get_config())

    with tf.device("/cpu:0"):
        cpu_model = tf.keras.models.clone_model(model, clone_function=clone_layer)
        cpu_model.build(model.input_shape)
        cpu_model.set_weights(model.get_weights())
    return cpu_model

def _save_tflite_int8_windows(model, X_windows, models_dir, stem):
    def representative_dataset():
        if len(X_windows) == 0:
            return
        idx = np.random.default_rng(42).choice(
            len(X_windows), min(200, len(X_windows)), replace=False
        )
        for i in idx:
            yield [X_windows[i:i + 1].astype(np.float32)]

    cpu_model = _clone_model_for_tflite(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(cpu_model)
    converter.optimizations              = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset     = representative_dataset
    converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type       = tf.int8
    converter.inference_output_type      = tf.int8
    tflite_bytes = _silent_tflite_convert(converter)

    tflite_path = models_dir / f"{stem}_int8.tflite"
    tflite_path.write_bytes(tflite_bytes)
    _export_c_array(tflite_bytes, models_dir / f"{stem}_int8.h", stem)
    print(f"  TFLite : {tflite_path.name}  ({len(tflite_bytes)/1024:.1f} KB / {STM32_FLASH_KB} KB)")
    del cpu_model; gc.collect()


def _save_tflite_int8(model, df_representative, window_size, step_size, models_dir, stem):
    X_repr, _ = extract_all_windows(df_representative, window_size, step_size)
    _save_tflite_int8_windows(model, X_repr, models_dir, stem)

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
        f"LSTM  |  Fold {fold_idx} — P{test_part} S{test_sess}"
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
# 6. DONNÉES — Générateurs tf.data.Dataset
# ══════════════════════════════════════════════════════════════════════
def get_window_indices(df, window_size, step_size):
    """
    Retourne les indices de début de chaque fenêtre pour chaque session.
    """
    indices = []
    groups = df.groupby([COL_PARTICIPANT, COL_SESSION])
    group_data = [] # Liste de (X, y)
    
    current_offset = 0
    full_X = df[SIGNAL_COLS].values.astype(np.float32)
    full_y = df[COL_LABEL].values.astype(np.int32)
    
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

    X_raw, y_raw, idxs = get_window_indices(df, window_size, step_size)
    if len(idxs) == 0:
        return None, 0

    def fetch_window(i):
        i = int(i)
        win_X = X_raw[i : i + window_size]
        # Label majoritaire
        win_y = y_raw[i : i + window_size]
        label = np.argmax(np.bincount(win_y, minlength=num_classes))
        return win_X, to_categorical(label, num_classes=num_classes)

    dataset = tf.data.Dataset.from_tensor_slices(idxs)
    dataset = dataset.map(
        lambda i: tf.py_function(fetch_window, [i], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # On définit les formes explicitement car tf.py_function les perd
    dataset = dataset.map(lambda x, y: (tf.ensure_shape(x, (window_size, len(SIGNAL_COLS))), 
                                       tf.ensure_shape(y, (num_classes,))))

    dataset = dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    steps = (len(idxs) + batch_size - 1) // batch_size
    return dataset, steps

def get_window_labels(df, window_size, step_size):
    """
    Version optimisée de l'extraction des labels.
    """
    y_all = []
    groups = df.groupby([COL_PARTICIPANT, COL_SESSION])
    for _, group in groups:
        y = group[COL_LABEL].values.astype(np.int32)
        for start in range(0, len(y) - window_size + 1, step_size):
            end = start + window_size
            counts = np.bincount(y[start:end])
            y_all.append(np.argmax(counts))
    return np.array(y_all)

# ══════════════════════════════════════════════════════════════════════
# 7. MODÈLE + OPTUNA
# ══════════════════════════════════════════════════════════════════════
def build_model(input_shape, num_classes, params):
    n_layers     = params.get("n_lstm_layers", 1)
    units_1      = params["lstm_units"]
    units_2      = params.get("lstm_units_2", max(16, units_1 // 2))
    use_bidir    = params.get("bidirectional", False)
    dropout_rate = params.get("dropout_rate", 0.4)
    l2_strength  = params.get("l2_reg", 0.0)
    dense_units  = params.get("dense_units", 64)
    lr           = params["learning_rate"]
    reg          = l2(l2_strength) if l2_strength > 0 else None

    def _make_lstm(units, return_seq):
        # stateful=False supprime la contrainte dure du batch padding et libère mémoire
        core = LSTM(units, stateful=False, return_sequences=return_seq,
                    dropout=dropout_rate, kernel_regularizer=reg)
        return Bidirectional(core) if use_bidir else core

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(_make_lstm(units_1, return_seq=(n_layers > 1)))
    model.add(BatchNormalization())
    if n_layers > 1:
        model.add(_make_lstm(units_2, return_seq=False))
        model.add(BatchNormalization())
    model.add(Dense(dense_units, activation="relu", kernel_regularizer=reg))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def optuna_objective(trial, df_splits, window_size, step_size, num_classes):
    n_lstm_layers = trial.suggest_int("n_lstm_layers", **LSTM_OPTUNA_SPACE["n_lstm_layers"])
    lstm_units    = trial.suggest_categorical("lstm_units",  LSTM_OPTUNA_SPACE["lstm_units"])
    lstm_units_2  = (trial.suggest_categorical("lstm_units_2", LSTM_OPTUNA_SPACE["lstm_units_2"])
                     if n_lstm_layers > 1 else 32)
    bidirectional = trial.suggest_categorical("bidirectional", LSTM_OPTUNA_SPACE["bidirectional"])
    dense_units   = trial.suggest_categorical("dense_units",   LSTM_OPTUNA_SPACE["dense_units"])
    dropout_rate  = trial.suggest_float("dropout_rate",  **LSTM_OPTUNA_SPACE["dropout_rate"])
    l2_reg        = trial.suggest_float("l2_reg",        **LSTM_OPTUNA_SPACE["l2_reg"])
    learning_rate = trial.suggest_float("learning_rate", **LSTM_OPTUNA_SPACE["learning_rate"])
    batch_size    = trial.suggest_categorical("batch_size", LSTM_OPTUNA_SPACE["batch_size"])

    params = {
        "n_lstm_layers": n_lstm_layers, "lstm_units": lstm_units,
        "lstm_units_2":  lstm_units_2,  "bidirectional": bidirectional,
        "dense_units":   dense_units,   "dropout_rate": dropout_rate,
        "l2_reg":        l2_reg,        "learning_rate": learning_rate,
        "batch_size":    batch_size,
    }
    scores = []
    
    for (df_fit, df_val) in df_splits:
        ds_fit, st_fit = create_tf_dataset(df_fit, window_size, step_size, batch_size, num_classes)
        ds_val, st_val = create_tf_dataset(df_val, window_size, step_size, batch_size, num_classes)
        
        if ds_fit is None or ds_val is None:
            scores.append(0.0)
            continue
            
        model = None
        try:
            model = build_model((window_size, len(SIGNAL_COLS)), num_classes, params)
            model.fit(
                ds_fit,
                validation_data=ds_val,
                epochs=15, 
                steps_per_epoch=st_fit,
                validation_steps=st_val,
                shuffle=False,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                    TerminateOnNaN()
                ],
                verbose=0,
            )
            
            y_pred = np.argmax(model.predict(ds_val, steps=st_val, verbose=0), axis=1)
            y_val_true = get_window_labels(df_val, window_size, step_size)
            
            min_len = min(len(y_pred), len(y_val_true))
            scores.append(f1_score(y_val_true[:min_len], y_pred[:min_len], average="macro", zero_division=0))
        except tf.errors.ResourceExhaustedError:
            raise optuna.exceptions.TrialPruned("OOM")
        except Exception as exc:
            print(f"  [WARN] split échoué ({type(exc).__name__}: {exc})")
            scores.append(0.0)
        finally:
            _free_memory(model)
            if 'ds_fit' in locals(): del ds_fit
            if 'ds_val' in locals(): del ds_val
            gc.collect()
            
    return float(np.mean(scores)) if scores else 0.0

def optimize_hyperparams(df, num_classes, n_trials=80):
    print(f"\nOPTUNA LSTM — {n_trials} trials | {N_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
    import random
    W_OPT = WINDOW_CONFIGS["default"]["window_size"]
    S_OPT = WINDOW_CONFIGS["default"]["step_size"]

    unique_sessions = [tuple(x) for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
    val_sessions    = random.sample(unique_sessions, min(N_OPTUNA_SESSIONS, len(unique_sessions)))
    print(f"Sessions : {val_sessions}")

    df_splits = []
    for (val_part, val_sess) in val_sessions:
        df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
        df_val   = df[ (df[COL_PARTICIPANT] == val_part)  & (df[COL_SESSION] == val_sess)].copy()
        scaler  = RobustScaler()
        df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
        df_val[SIGNAL_COLS]   = scaler.transform(df_val[SIGNAL_COLS])
        
        df_splits.append((df_train, df_val))
        
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study  = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )
    study.optimize(
        lambda trial: optuna_objective(trial, df_splits, W_OPT, S_OPT, num_classes),
        n_trials=n_trials, show_progress_bar=True,
        n_jobs=1, gc_after_trial=True, catch=(Exception,),
    )
    
    del df_splits; gc.collect()

    print(f"\nBest F1-Macro : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    OPTUNA_PATH_LSTM.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_LSTM, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return study.best_params, val_sessions

# ══════════════════════════════════════════════════════════════════════
# 8. MODÈLE GLOBAL
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

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]
    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]
    predictions = []

    for sample in X_data:
        x = sample[np.newaxis, ...].astype(np.float32)
        if input_dtype in (np.int8, np.uint8):
            x = np.round(x / input_scale + input_zero_point)
            x = np.clip(x, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
        else:
            x = x.astype(input_dtype)

        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details["index"])
        if output_dtype in (np.int8, np.uint8):
            y = (y.astype(np.float32) - output_zero_point) * output_scale
        predictions.append(int(np.argmax(y, axis=1)[0]))

    return np.array(predictions)

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
    print(f"  Comparaison edge → {METRICS_PATH}")

def _apply_pruning_and_finetune(model, X_tr, y_tr, X_vl, y_vl, weight_dict,
                                final_sparsity=0.5, pruning_epochs=20, batch_size=32):
    num_classes = model.output_shape[-1]
    steps_per_epoch = max(1, len(X_tr) // batch_size)
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=final_sparsity,
        begin_step=0,
        end_step=steps_per_epoch * pruning_epochs,
    )

    def _apply_pruning_to_dense(layer):
        # Seules les couches Dense sont prunables ici : LSTM/Bidirectional
        # ne sont pas supportées par tfmot.sparsity.keras.prune_low_magnitude.
        if isinstance(layer, Dense):
            return tfmot.sparsity.keras.prune_low_magnitude(
                layer, pruning_schedule=pruning_schedule
            )
        return layer

    model_prunable = tf.keras.models.clone_model(
        model, clone_function=_apply_pruning_to_dense
    )

    def _is_pruning_wrapper(layer):
        return (
            layer.__class__.__name__ == "PruneLowMagnitude"
            and hasattr(layer, "layer")
        )

    # Transfert des poids pré-entraînés couche par couche. Les couches
    # wrappées en PruneLowMagnitude exposent l'original via .layer.
    for orig_layer, cloned_layer in zip(model.layers, model_prunable.layers):
        if _is_pruning_wrapper(cloned_layer):
            cloned_layer.layer.set_weights(orig_layer.get_weights())
        else:
            cloned_layer.set_weights(orig_layer.get_weights())

    optimizer = tf.keras.optimizers.deserialize(
        tf.keras.optimizers.serialize(model.optimizer)
    )
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
        callbacks=[
            tfmot.sparsity.keras.UpdatePruningStep(),
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            TerminateOnNaN(),
        ],
        class_weight=weight_dict,
        verbose=1,
    )
    model_stripped = tfmot.sparsity.keras.strip_pruning(model_prunable)
    _free_memory(model_prunable)
    return model_stripped

def train_global_model(df_labeled, df_unlabeled, best_params, num_classes, val_sessions):
    print("\n" + "=" * 60 + f"\nMODÈLE GLOBAL — {MODEL_NAME}\n" + "=" * 60)
    config = WINDOW_CONFIGS["default"]
    W_SIZE, S_SIZE, EPOCHS = config["window_size"], config["step_size"], config["epochs"]
    bs = best_params.get("batch_size", 32)

    df_all = df_labeled.copy()
    scaler = RobustScaler()
    df_all[SIGNAL_COLS] = scaler.fit_transform(df_all[SIGNAL_COLS])
    df_unl = df_unlabeled.copy()
    if len(df_unl) > 0:
        df_unl[SIGNAL_COLS] = scaler.transform(df_unl[SIGNAL_COLS])

    val_sessions_set = set(val_sessions)
    mask_val = df_all.set_index([COL_PARTICIPANT, COL_SESSION]).index.isin(val_sessions_set)
    df_train = df_all[~mask_val].copy()
    df_val = df_all[mask_val].copy()

    X_tr, y_tr = extract_all_windows(df_train, W_SIZE, S_SIZE)
    X_vl, y_vl = extract_all_windows(df_val, W_SIZE, S_SIZE)
    X_unlabeled, _ = (
        extract_all_windows(df_unl, W_SIZE, S_SIZE)
        if len(df_unl) > 0 else (np.array([]), np.array([]))
    )
    del df_all, df_unl, df_train, df_val
    gc.collect()
    print(f"  Fenêtres train : {len(X_tr)} | val : {len(X_vl)} | unlabeled : {len(X_unlabeled)}")

    w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    weight_dict = dict(zip(np.unique(y_tr), w))

    model = None
    try:
        model = build_model((W_SIZE, len(SIGNAL_COLS)), num_classes, best_params)
        model.fit(
            X_tr, to_categorical(y_tr, num_classes),
            validation_data=(X_vl, to_categorical(y_vl, num_classes)),
            epochs=EPOCHS,
            batch_size=bs,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                TerminateOnNaN(),
            ],
            class_weight=weight_dict,
            verbose=1,
        )

        X_tr_v2, y_tr_v2, pseudo_y, _ = add_pseudo_labels(model, X_tr, y_tr, X_unlabeled)
        if len(pseudo_y) > 0:
            w_v2 = compute_class_weight("balanced", classes=np.unique(y_tr_v2), y=y_tr_v2)
            weight_dict_v2 = dict(zip(np.unique(y_tr_v2), w_v2))
            _free_memory(model)
            model = build_model((W_SIZE, len(SIGNAL_COLS)), num_classes, best_params)
            model.fit(
                X_tr_v2, to_categorical(y_tr_v2, num_classes),
                validation_data=(X_vl, to_categorical(y_vl, num_classes)),
                epochs=EPOCHS,
                batch_size=bs,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                    TerminateOnNaN(),
                ],
                class_weight=weight_dict_v2,
                verbose=1,
            )
            X_repr = np.concatenate([X_tr_v2, X_vl], axis=0)
            X_tr_final, y_tr_final, weight_dict_final = X_tr_v2, y_tr_v2, weight_dict_v2
        else:
            X_repr = np.concatenate([X_tr, X_vl], axis=0)
            X_tr_final, y_tr_final, weight_dict_final = X_tr, y_tr, weight_dict

        models_dir = MODELS_DIR / "LSTM"
        models_dir.mkdir(parents=True, exist_ok=True)

        # --- Étape 1 : modèle natif NON prunné → .keras + métriques "float32" ---
        float32_path = models_dir / f"{MODEL_NAME}_global_float32.keras"
        model.save(float32_path)
        y_pred_float32 = np.argmax(model.predict(X_vl, verbose=0), axis=1)
        float32_metrics = _compute_classification_metrics(y_vl, y_pred_float32)
        float32_metrics["model_size_kb"] = float(float32_path.stat().st_size / 1024)

        # --- Étape 2 : pruning (Dense uniquement), avec repli propre en cas d'échec ---
        model_for_export = model
        try:
            print("  Pruning magnitude (Dense uniquement) : fine-tuning...")
            model_for_export = _apply_pruning_and_finetune(
                model, X_tr_final, y_tr_final, X_vl, y_vl, weight_dict_final,
                batch_size=bs,
            )
            print("  Pruning magnitude appliqué avec succès.")
        except Exception as exc:
            print(f"  Pruning magnitude échoué ({type(exc).__name__}: {exc}) — repli sur le modèle non prunné pour l'export TFLite.")
            traceback.print_exc()
            model_for_export = model

        # --- Étape 3 : export TFLite INT8 du modèle prunné (ou du repli) → .tflite + .h ---
        _save_tflite_int8_windows(model_for_export, X_repr, models_dir, f"{MODEL_NAME}_global")
        tflite_path = models_dir / f"{MODEL_NAME}_global_int8.tflite"
        y_pred_tflite = _predict_tflite(tflite_path, X_vl)
        tflite_metrics = _compute_classification_metrics(y_vl, y_pred_tflite)
        tflite_metrics["model_size_kb"] = float(tflite_path.stat().st_size / 1024)

        edge_metrics = {
            "float32": float32_metrics,
            "int8_tflite": tflite_metrics,
        }
        _save_edge_comparison_metrics(MODEL_NAME, edge_metrics)
        print(f"  Modèle global exporté (.keras + .tflite + .h).")
        print(f"  Comparaison edge sauvegardée dans {METRICS_PATH}")

        if model_for_export is not model:
            _free_memory(model_for_export)
    except Exception as exc:
        print(f"  Modèle global échoué ({type(exc).__name__}: {exc})")
    finally:
        _free_memory(model)
        for name in ("X_tr", "X_vl", "X_unlabeled", "X_repr"):
            if name in locals():
                del locals()[name]
        gc.collect()

# ══════════════════════════════════════════════════════════════════════
# 9. BOUCLE LOSO
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

    if USE_OPTUNA_LSTM:
        best_params, val_sessions = optimize_hyperparams(df_labeled, num_classes)
    else:
        best_params = MODEL_PARAMS["LSTM"]
        all_sess = [tuple(x) for x in df_labeled[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
        import random as _random
        val_sessions = _random.sample(all_sess, min(N_OPTUNA_SESSIONS, len(all_sess)))

    all_metrics = []
    import random
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

        bs = best_params.get("batch_size", 32)

        ds_fit,  st_fit  = create_tf_dataset(df_fit,  W_SIZE, S_SIZE, bs, num_classes)
        ds_val,  st_val  = create_tf_dataset(df_val,  W_SIZE, S_SIZE, bs, num_classes)
        ds_test, st_test = create_tf_dataset(df_test, W_SIZE, S_SIZE, bs, num_classes)

        if ds_test is None:
            print(f"  WARNING: test vide pour P{test_part}. Skip."); continue
        if ds_val is None or ds_fit is None:
            print(f"  WARNING: Fit/Val vide. Skip."); continue

        y_fitLabels = get_window_labels(df_fit, W_SIZE, S_SIZE)
        if len(y_fitLabels) == 0:
             continue
        w = compute_class_weight("balanced", classes=np.unique(y_fitLabels), y=y_fitLabels)
        weight_dict = dict(enumerate(w))

        model = None
        try:
            model = build_model((W_SIZE, len(SIGNAL_COLS)), num_classes, best_params)
            history = model.fit(
                ds_fit,
                epochs=EPOCHS,
                validation_data=ds_val,
                steps_per_epoch=st_fit,
                validation_steps=st_val,
                shuffle=False, 
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                    TerminateOnNaN()
                ],
                class_weight=weight_dict,
                verbose=1,
            )

            X_fit_v2, y_fit_v2, pseudo_y, _ = add_pseudo_labels(model, X_fit, y_fit, X_unlabeled)
            if len(pseudo_y) > 0:
                w_v2 = compute_class_weight("balanced", classes=np.unique(y_fit_v2), y=y_fit_v2)
                weight_dict_v2 = dict(zip(np.unique(y_fit_v2), w_v2))
                _free_memory(model)
                model = build_model((W_SIZE, len(SIGNAL_COLS)), num_classes, best_params)
                history = model.fit(
                    X_fit_v2, to_categorical(y_fit_v2, num_classes),
                    epochs=EPOCHS,
                    validation_data=(X_val_np, to_categorical(y_val_np, num_classes)),
                    batch_size=bs,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                        TerminateOnNaN(),
                    ],
                    class_weight=weight_dict_v2,
                    verbose=1,
                )

            y_pred  = np.argmax(model.predict(ds_test, steps=st_test, verbose=0), axis=1)
            y_true  = get_window_labels(df_test, W_SIZE, S_SIZE)
            
            min_len = min(len(y_pred), len(y_true))
            y_pred  = y_pred[:min_len]
            y_true  = y_true[:min_len]

            f1_mac  = f1_score(y_true, y_pred, average="macro")
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            print(f"  ✅ F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            plots_dir = BASE_DIR / "training_curves" / "LSTM"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_fold(history, test_idx+1, test_part, test_sess,
                      plots_dir, f1_mac, bal_acc, y_true, y_pred)

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

    train_global_model(df_labeled, df_unlabeled, best_params, num_classes, val_sessions)


if __name__ == "__main__":
    main()
