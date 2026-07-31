import gc
import json
import os
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
# pyrefly: ignore [missing-import]
import matplotlib; matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf

# pyrefly: ignore [missing-import]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *

from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (
    BatchNormalization, Conv1D, Dense, Dropout,
    Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D, Input, MaxPooling1D,)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from models.semi_supervised import add_pseudo_labels, extract_all_windows , extract_windows
from utils.apply_smote import resample_dataframe

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ══════════════════════════════════════════════════════════════════════
# 2. CONSTANTES
# ══════════════════════════════════════════════════════════════════════
MODEL_NAME        = "CNN_1D"
LABEL_MAPPING     = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES      = ["baseline", "activity", "pre_fatigue", "fatigue"]


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
    print(f"  TFLite : {tflite_path.name}  ({len(tflite_bytes)/1024:.1f} KB / {STM32_FLASH_KB} KB)")

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
    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]
    input_scale, input_zero_point = input_details.get("quantization", (0.0, 0))
    output_scale, output_zero_point = output_details.get("quantization", (0.0, 0))

    predictions = []
    for sample in X_data:
        input_data = sample[np.newaxis, ...].astype(np.float32)
        if np.issubdtype(input_dtype, np.integer):
            if input_scale == 0:
                raise ValueError(f"Quantification d'entrée invalide pour {tflite_path}")
            q_info = np.iinfo(input_dtype)
            input_data = np.round(input_data / input_scale + input_zero_point)
            input_data = np.clip(input_data, q_info.min, q_info.max).astype(input_dtype)
        else:
            input_data = input_data.astype(input_dtype)

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)
        if np.issubdtype(output_dtype, np.integer):
            if output_scale == 0:
                raise ValueError(f"Quantification de sortie invalide pour {tflite_path}")
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        predictions.append(int(np.argmax(output_data, axis=1)[0]))

    return np.array(predictions)

def _save_edge_comparison_metrics(model_name, edge_metrics):
    curr_metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                content = f.read().strip()
            curr_metrics = json.loads(content) if content else {}
        except json.JSONDecodeError:
            print(f"metrics.json invalide ou vide, réinitialisation : {METRICS_PATH}")
            curr_metrics = {}

    curr_metrics.setdefault(model_name, {})
    curr_metrics[model_name]["edge_comparison"] = edge_metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(curr_metrics, f, indent=4)

def _apply_pruning_and_finetune(model, X_tr, y_tr, X_vl, y_vl, weight_dict,
                                final_sparsity=0.5, pruning_epochs=20, batch_size=32):
    steps_per_epoch = max(1, len(X_tr) // batch_size)
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=final_sparsity,
        begin_step=0,
        end_step=steps_per_epoch * pruning_epochs,
    )
    model_prunable = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        pruning_schedule=pruning_schedule,
    )
    optimizer = tf.keras.optimizers.deserialize(
        tf.keras.optimizers.serialize(model.optimizer)
    )
    model_prunable.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    model_prunable.fit(
        X_tr, to_categorical(y_tr, model.output_shape[-1]),
        validation_data=(X_vl, to_categorical(y_vl, model.output_shape[-1])),
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
        f"CNN-1D  |  Fold {fold_idx} — P{test_part} S{test_sess}"
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
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
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
# 7. MODÈLE + OPTUNA
# ══════════════════════════════════════════════════════════════════════
def build_model(trial, input_shape, num_classes):
    n_conv_blocks  = trial.suggest_categorical("n_conv_blocks", CNN_D1_OPTUNA_SPACE["n_conv_blocks"])
    kernel_size    = trial.suggest_categorical("kernel_size",   CNN_D1_OPTUNA_SPACE["kernel_size"])
    filters        = trial.suggest_categorical("filters",       CNN_D1_OPTUNA_SPACE["filters"])
    use_batchnorm  = trial.suggest_categorical("use_batchnorm", CNN_D1_OPTUNA_SPACE["use_batchnorm"])
    activation     = trial.suggest_categorical("activation",    CNN_D1_OPTUNA_SPACE["activation"])
    pool_size      = trial.suggest_categorical("pool_size",     CNN_D1_OPTUNA_SPACE["pool_size"])
    global_pooling = trial.suggest_categorical("global_pooling",CNN_D1_OPTUNA_SPACE["global_pooling"])
    l2_reg         = trial.suggest_float("l2_reg",       *CNN_D1_OPTUNA_SPACE["l2_reg"], log=True)
    optimizer_str  = trial.suggest_categorical("optimizer",     CNN_D1_OPTUNA_SPACE["optimizer"])
    dense_units    = trial.suggest_categorical("dense_units",   CNN_D1_OPTUNA_SPACE["dense_units"])
    dropout_rate   = trial.suggest_float("dropout_rate",  *CNN_D1_OPTUNA_SPACE["dropout_rate"])
    learning_rate  = trial.suggest_float("learning_rate", *CNN_D1_OPTUNA_SPACE["learning_rate"], log=True)

    layers = [Input(shape=input_shape)]
    for i in range(n_conv_blocks):
        layers.append(Conv1D(filters, kernel_size, activation=activation, kernel_regularizer=l2(l2_reg)))
        if use_batchnorm:
            layers.append(BatchNormalization())
        if input_shape[0] // (pool_size ** (i + 1)) > 1:
            layers.append(MaxPooling1D(pool_size=pool_size))

    if   global_pooling == "flatten": layers.append(Flatten())
    elif global_pooling == "avg":     layers.append(GlobalAveragePooling1D())
    else:                             layers.append(GlobalMaxPooling1D())

    layers += [
        Dense(dense_units, activation=activation, kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax"),
    ]
    model = Sequential(layers)
    opt = Adam(learning_rate) if optimizer_str == "adam" else RMSprop(learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=["accuracy"], jit_compile=False)
    return model

def optuna_objective(trial, df, num_classes):
    config      = WINDOW_CONFIGS["default"]
    window_size = config["window_size"]
    step_size   = config["step_size"]
    batch_size  = trial.suggest_categorical("batch_size", CNN_D1_OPTUNA_SPACE["batch_size"])

    import random
    unique_sessions = [tuple(x) for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
    val_sessions    = random.sample(unique_sessions, min(CNN_D1_OPTUNA_SESSIONS, len(unique_sessions)))
    scores = []

    for fold_idx, (val_part, val_sess) in enumerate(val_sessions):
        model = None
        try:
            df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
            df_val   = df[ (df[COL_PARTICIPANT] == val_part)  & (df[COL_SESSION] == val_sess)].copy()

            # SMOTE seulement sur df_train
            df_train = resample_dataframe(df_train, SIGNAL_COLS)

            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_val[SIGNAL_COLS]   = scaler.transform(df_val[SIGNAL_COLS])

            X_train, y_train = extract_all_windows(df_train, window_size, step_size)
            X_val,   y_val   = extract_all_windows(df_val,   window_size, step_size)
            if len(X_val) == 0: continue
            w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), w))

            model = build_model(trial, (window_size, len(SIGNAL_COLS)), num_classes)
            model.fit(
                X_train, to_categorical(y_train, num_classes),
                epochs=10, batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=[TerminateOnNaN()], verbose=0,
            )
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            score  = f1_score(y_val, y_pred, average="macro", zero_division=0)
            scores.append(score)

            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        except tf.errors.ResourceExhaustedError:
            raise optuna.exceptions.TrialPruned("OOM")
        except Exception as exc:
            print(f"  [WARN] fold échoué ({type(exc).__name__}: {exc})")
            scores.append(0.0)
        finally:
            _free_memory(model)
            if 'X_train' in locals(): del X_train
            if 'X_val'   in locals(): del X_val
            if 'y_train' in locals(): del y_train
            if 'y_val'   in locals(): del y_val

    return float(np.mean(scores)) if scores else 0.0

def optimize_hyperparams(df, num_classes):
    print(f"\nOPTUNA CNN-1D — {CNN_D1_OPTUNA_TRIALS} trials | {CNN_D1_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, df, num_classes),
        n_trials=CNN_D1_OPTUNA_TRIALS, show_progress_bar=True,
        gc_after_trial=True, catch=(Exception,),
    )
    print(f"\nBest F1-Macro : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    OPTUNA_PATH_CNN_D1.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_CNN_D1, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return study.best_params

# ══════════════════════════════════════════════════════════════════════
# 9. MODÈLE GLOBAL
# ══════════════════════════════════════════════════════════════════════
def train_global_model(df_labeled, df_unlabeled, best_params, num_classes):
    print("\n" + "=" * 60 + f"\nMODÈLE GLOBAL — {MODEL_NAME}\n" + "=" * 60)
    config = WINDOW_CONFIGS["default"]
    W_SIZE, S_SIZE, EPOCHS = config["window_size"], config["step_size"], config["epochs"]

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
    X_repr = X_all
    try:
        idx = np.random.default_rng(42).permutation(len(X_all))
        split = int(0.9 * len(X_all))
        X_tr, y_tr = X_all[idx[:split]], y_all[idx[:split]]
        X_vl, y_vl = X_all[idx[split:]], y_all[idx[split:]]
        model = build_model(
            optuna.trial.FixedTrial(best_params),
            (W_SIZE, len(SIGNAL_COLS)), num_classes
        )
        w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        weight_dict = dict(zip(np.unique(y_tr), w))
        model.fit(
            X_tr, to_categorical(y_tr, num_classes),
            validation_data=(X_vl, to_categorical(y_vl, num_classes)),
            epochs=EPOCHS,
            batch_size=best_params.get("batch_size", 32),
            callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                       TerminateOnNaN()],
            class_weight=weight_dict,
            verbose=1,
        )

        X_tr_v2, y_tr_v2, pseudo_y, _ = add_pseudo_labels(model, X_tr, y_tr, X_unlabeled)
        if len(pseudo_y) > 0:
            w_v2 = compute_class_weight("balanced", classes=np.unique(y_tr_v2), y=y_tr_v2)
            weight_dict_v2 = dict(zip(np.unique(y_tr_v2), w_v2))
            _free_memory(model)
            model = build_model(
                optuna.trial.FixedTrial(best_params),
                (W_SIZE, len(SIGNAL_COLS)), num_classes
            )
            model.fit(
                X_tr_v2, to_categorical(y_tr_v2, num_classes),
                validation_data=(X_vl, to_categorical(y_vl, num_classes)),
                epochs=EPOCHS,
                batch_size=best_params.get("batch_size", 32),
                callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                           TerminateOnNaN()],
                class_weight=weight_dict_v2,
                verbose=1,
            )
            X_repr = np.concatenate([X_tr_v2, X_vl], axis=0)
        else:
            X_repr = X_all

        models_dir = MODELS_DIR / "CNN"
        models_dir.mkdir(parents=True, exist_ok=True)
        float32_path = models_dir / f"{MODEL_NAME}_global_float32.keras"
        model.save(float32_path)
        y_pred_float32 = np.argmax(model.predict(X_vl, verbose=0), axis=1)
        float32_metrics = _compute_classification_metrics(y_vl, y_pred_float32)
        float32_metrics["model_size_kb"] = float(float32_path.stat().st_size / 1024)

        _save_tflite_int8(model, X_repr, models_dir, f"{MODEL_NAME}_global")
        tflite_path = models_dir / f"{MODEL_NAME}_global_int8.tflite"
        y_pred_tflite = _predict_tflite(tflite_path, X_vl)
        tflite_metrics = _compute_classification_metrics(y_vl, y_pred_tflite)
        tflite_metrics["model_size_kb"] = float(tflite_path.stat().st_size / 1024)

        edge_metrics = {
            "float32": float32_metrics,
            "int8_tflite": tflite_metrics,
        }

        model_pruned = None
        try:
            print("  Pruning magnitude : fine-tuning du modèle global...")
            model_pruned = _apply_pruning_and_finetune(
                model, X_tr, y_tr, X_vl, y_vl, weight_dict,
                batch_size=best_params.get("batch_size", 32),
            )
            pruned_float32_path = models_dir / f"{MODEL_NAME}_global_pruned_float32.keras"
            model_pruned.save(pruned_float32_path)
            y_pred_pruned = np.argmax(model_pruned.predict(X_vl, verbose=0), axis=1)
            pruned_metrics = _compute_classification_metrics(y_vl, y_pred_pruned)
            pruned_metrics["model_size_kb"] = float(pruned_float32_path.stat().st_size / 1024)
            edge_metrics["pruned_float32"] = pruned_metrics

            _save_tflite_int8(model_pruned, X_repr, models_dir, f"{MODEL_NAME}_global_pruned")
            pruned_tflite_path = models_dir / f"{MODEL_NAME}_global_pruned_int8.tflite"
            y_pred_pruned_tflite = _predict_tflite(pruned_tflite_path, X_vl)
            pruned_tflite_metrics = _compute_classification_metrics(y_vl, y_pred_pruned_tflite)
            pruned_tflite_metrics["model_size_kb"] = float(pruned_tflite_path.stat().st_size / 1024)
            edge_metrics["pruned_int8_tflite"] = pruned_tflite_metrics
            print("  Pruning magnitude : exports et métriques sauvegardés.")
        except Exception as exc:
            print(f"  Pruning magnitude échoué ({type(exc).__name__}: {exc})")
            traceback.print_exc()
        finally:
            _free_memory(model_pruned)

        _save_edge_comparison_metrics(MODEL_NAME, edge_metrics)
        print(f"  Modèle global exporté (.keras + .tflite + .h).")
        print(f"  Comparaison edge sauvegardée dans {METRICS_PATH}")
    except Exception as exc:
        print(f"  Modèle global échoué ({type(exc).__name__}: {exc})")
        traceback.print_exc()
    finally:
        _free_memory(model)
        del X_all, y_all
        gc.collect()

# ══════════════════════════════════════════════════════════════════════
# 8. BOUCLE LOSO
# ══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60 + f"\nTRAIN LOSO — {MODEL_NAME}\n" + "=" * 60)
    if not DATA_MODEL_READY.exists():
        return print(f" Dataset non trouvé : {DATA_MODEL_READY}")

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

    best_params = (
        optimize_hyperparams(df_labeled, num_classes)
        if USE_OPTUNA_CNN_1D
        else MODEL_PARAMS["CNN_1D"]
    )

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


        # SMOTE seulement sur df_train
        df_fit = resample_dataframe(df_fit, SIGNAL_COLS)

        scaler = RobustScaler()
        df_fit[SIGNAL_COLS]  = scaler.fit_transform(df_fit[SIGNAL_COLS])
        df_val[SIGNAL_COLS]  = scaler.transform(df_val[SIGNAL_COLS])
        df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])
        df_unlabeled_fit = df_unlabeled.copy()
        if len(df_unlabeled_fit) > 0:
            df_unlabeled_fit[SIGNAL_COLS] = scaler.transform(df_unlabeled_fit[SIGNAL_COLS])


        X_fit,  y_fit  = extract_all_windows(df_fit,  W_SIZE, S_SIZE)
        X_val,  y_val  = extract_all_windows(df_val,  W_SIZE, S_SIZE)
        X_test, y_test = extract_all_windows(df_test, W_SIZE, S_SIZE)
        X_unlabeled, _ = extract_all_windows(df_unlabeled_fit, W_SIZE, S_SIZE) if len(df_unlabeled_fit) > 0 else (np.array([]), np.array([]))

        if len(X_test) == 0:
            print(f"  WARNING: test vide pour P{test_part}. Skip."); continue
        if len(X_val) == 0:
            print(f"  WARNING: val vide pour P{val_part}. Skip.");   continue

        print(f"  Fit: {len(X_fit)} | Val: {len(X_val)} | Test: {len(X_test)} fenêtres")
        w           = compute_class_weight("balanced", classes=np.unique(y_fit), y=y_fit)
        weight_dict = dict(zip(np.unique(y_fit), w))

        model = None
        try:
            model = build_model(
                optuna.trial.FixedTrial(best_params),
                (W_SIZE, len(SIGNAL_COLS)), num_classes
            )
            history = model.fit(
                X_fit, to_categorical(y_fit, num_classes),
                epochs=EPOCHS,
                validation_data=(X_val, to_categorical(y_val, num_classes)),
                batch_size=best_params.get("batch_size", 32),
                callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                           TerminateOnNaN()],
                class_weight=weight_dict,
                verbose=1,
            )

            X_fit_v2, y_fit_v2, pseudo_y, _ = add_pseudo_labels(model, X_fit, y_fit, X_unlabeled)
            if len(pseudo_y) > 0:
                w_v2 = compute_class_weight("balanced", classes=np.unique(y_fit_v2), y=y_fit_v2)
                weight_dict_v2 = dict(zip(np.unique(y_fit_v2), w_v2))
                _free_memory(model)
                model = build_model(
                    optuna.trial.FixedTrial(best_params),
                    (W_SIZE, len(SIGNAL_COLS)), num_classes
                )
                history = model.fit(
                    X_fit_v2, to_categorical(y_fit_v2, num_classes),
                    epochs=EPOCHS,
                    validation_data=(X_val, to_categorical(y_val, num_classes)),
                    batch_size=best_params.get("batch_size", 32),
                    callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                               TerminateOnNaN()],
                    class_weight=weight_dict_v2,
                    verbose=1,
                )

            y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
            f1_mac  = f1_score(y_test, y_pred, average="macro")
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            plots_dir = BASE_DIR / "training_curves" / "CNN"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_fold(history, test_idx+1, test_part, test_sess,
                      plots_dir, f1_mac, bal_acc, y_test, y_pred)

            all_metrics.append({
                "Fold": test_idx+1, "Participant": int(test_part), "Session": int(test_sess),
                "F1_Macro": float(f1_mac), "Balanced_Accuracy": float(bal_acc),
            })

        except Exception as exc:
            print(f"Fold {test_idx+1} échoué ({type(exc).__name__}: {exc})")
        finally:
            _free_memory(model)
            del X_fit, X_val, X_test, y_fit, y_val, y_test
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
        except json.JSONDecodeError:
            print(f"metrics.json invalide ou vide, réinitialisation : {METRICS_PATH}")
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

    train_global_model(df_labeled, df_unlabeled, best_params, num_classes)


if __name__ == "__main__":
    main()
