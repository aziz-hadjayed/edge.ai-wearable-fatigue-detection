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
import tensorflow_model_optimization as tfmot

# Supprimer la verbosité de SavedModel lors de la conversion TFLite
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
    Dense, Dropout, GlobalAveragePooling1D, Input, MaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from utils.apply_smote import resample_dataframe ,resample_windows_smote

# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from models.semi_supervised import add_pseudo_labels, extract_all_windows , extract_windows

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ══════════════════════════════════════════════════════════════════════
# 2. CONSTANTES
# ══════════════════════════════════════════════════════════════════════
MODEL_NAME        = "CNN_TCN"
LABEL_MAPPING     = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES      = ["baseline", "activity", "pre_fatigue", "fatigue"]

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
        f"CNN-TCN  |  Fold {fold_idx} — P{test_part} S{test_sess}"
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
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
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

def precompute_session_windows(df, window_size, step_size):
    """Pré-extrait les fenêtres brutes (float16) par session — une seule fois pour Optuna."""
    session_windows = {}
    if COL_TIMESTAMP in df.columns:
        df = df.sort_values([COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP])
    for (part, sess), group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        wx, wy = extract_windows(group, window_size, step_size, SIGNAL_COLS, COL_LABEL)
        if wx:
            session_windows[(part, sess)] = (np.array(wx, dtype=np.float16), np.array(wy))
    total_mb = sum(x.nbytes for x, _ in session_windows.values()) / 1024 ** 2
    print(f"  Fenêtres pré-calculées : {len(session_windows)} sessions — {total_mb:.1f} MB RAM")
    return session_windows

# ══════════════════════════════════════════════════════════════════════
# 7. MODÈLE + OPTUNA
# ══════════════════════════════════════════════════════════════════════
def _act(name, alpha=0.1):
    from tensorflow.keras.layers import LeakyReLU
    from tensorflow.keras.layers import Activation as Act
    return LeakyReLU(alpha=alpha) if name == "leaky_relu" else Act(name)

def _tcn_residual_block(x, filters, kernel_size, dilation, dropout_rate, activation_name, l2_reg):
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
    Architecture CNN-TCN :
      CNN Block  : 2× Conv1D causale → BN → Act → MaxPool
      TCN Stack  : N blocs dilatés (dilation=1,2,4,8…) avec connexion résiduelle
      Tête       : GAP → Dense → Dropout → Softmax(3)
    """
    cnn_filters   = trial.suggest_categorical("cnn_filters",  CNN_TCN_OPTUNA_SPACE["cnn_filters"])
    cnn_kernel    = trial.suggest_categorical("cnn_kernel",   CNN_TCN_OPTUNA_SPACE["cnn_kernel"])
    n_tcn_blocks  = trial.suggest_categorical("n_tcn_blocks", CNN_TCN_OPTUNA_SPACE["n_tcn_blocks"])
    tcn_filters   = trial.suggest_categorical("tcn_filters",  CNN_TCN_OPTUNA_SPACE["tcn_filters"])
    tcn_kernel    = trial.suggest_categorical("tcn_kernel",   CNN_TCN_OPTUNA_SPACE["tcn_kernel"])
    activation    = trial.suggest_categorical("activation",   CNN_TCN_OPTUNA_SPACE["activation"])
    l2_reg        = trial.suggest_float("l2_reg",       *CNN_TCN_OPTUNA_SPACE["l2_reg"], log=True)
    optimizer_str = trial.suggest_categorical("optimizer",    CNN_TCN_OPTUNA_SPACE["optimizer"])
    dense_units   = trial.suggest_categorical("dense_units",  CNN_TCN_OPTUNA_SPACE["dense_units"])
    dropout_rate  = trial.suggest_float("dropout_rate",  *CNN_TCN_OPTUNA_SPACE["dropout_rate"])
    learning_rate = trial.suggest_float("learning_rate", *CNN_TCN_OPTUNA_SPACE["learning_rate"], log=True)

    inputs = Input(shape=input_shape)
    x = Conv1D(cnn_filters, cnn_kernel, padding="causal", use_bias=False,
               kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv1D(cnn_filters, cnn_kernel, padding="causal", use_bias=False,
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPooling1D(pool_size=2)(x)

    for i in range(n_tcn_blocks):
        x = _tcn_residual_block(x, tcn_filters, tcn_kernel,
                                dilation=2**i, dropout_rate=dropout_rate,
                                activation_name=activation, l2_reg=l2_reg)

    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation=activation, kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="CNN_TCN")
    opt = Adam(learning_rate) if optimizer_str == "adam" else RMSprop(learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=["accuracy"], jit_compile=False)
    return model

def optuna_objective(trial, session_windows, num_classes, n_optuna_epochs=5):
    config      = WINDOW_CONFIGS["default"]
    window_size = config["window_size"]
    input_shape = (window_size, len(SIGNAL_COLS))
    batch_size  = trial.suggest_categorical("batch_size", CNN_TCN_OPTUNA_SPACE["batch_size"])

    import random
    all_sessions  = list(session_windows.keys())
    eval_sessions = random.sample(all_sessions, min(CNN_D1_TCN_OPTUNA_SESSIONS, len(all_sessions)))
    scores = []

    for fold_idx, (val_part, val_sess) in enumerate(eval_sessions):
        train_keys   = [(p, s) for (p, s) in all_sessions if (p, s) != (val_part, val_sess)]
        all_train_2d = np.concatenate(
            [session_windows[(p, s)][0].reshape(-1, len(SIGNAL_COLS)).astype(np.float32)
             for (p, s) in train_keys]
        )
        y_train = np.concatenate([session_windows[(p, s)][1] for (p, s) in train_keys])

        scaler  = RobustScaler().fit(all_train_2d)
        del all_train_2d; gc.collect()

        X_train = np.concatenate([
            scaler.transform(
                session_windows[(p, s)][0].reshape(-1, len(SIGNAL_COLS)).astype(np.float32)
            ).reshape(-1, window_size, len(SIGNAL_COLS))
            for (p, s) in train_keys
        ])
        X_val_raw, y_val = session_windows[(val_part, val_sess)]
        if len(X_val_raw) == 0:
            continue
        X_val = scaler.transform(
            X_val_raw.astype(np.float32).reshape(-1, len(SIGNAL_COLS))
        ).reshape(-1, window_size, len(SIGNAL_COLS))
        del scaler; gc.collect()

        X_train, y_train = resample_windows_smote(X_train, y_train)


        w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {c: wt for c, wt in zip(np.unique(y_train), w)}

        model = None
        try:
            _free_memory()
            model = build_model(trial, input_shape, num_classes)

            model_kb = model.count_params() / 1024
            if model_kb > EDGE_FLASH_BUDGET_KB:
                raise optuna.exceptions.TrialPruned(f"Flash {model_kb:.0f} KB > {EDGE_FLASH_BUDGET_KB} KB")

            model.fit(
                X_train, to_categorical(y_train, num_classes),
                epochs=n_optuna_epochs, batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=[TerminateOnNaN()], verbose=0,
            )
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            score  = f1_score(y_val, y_pred, average="macro", zero_division=0)
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
            if 'X_train' in locals(): del X_train
            if 'X_val'   in locals(): del X_val
            if 'y_train' in locals(): del y_train
            if 'y_val'   in locals(): del y_val
            gc.collect()

    return float(np.mean(scores)) if scores else 0.0

def optimize_hyperparams(df, num_classes, n_trials=CNN_D1_TCN_OPTUNA_TRIALS, n_optuna_epochs=10):
    print(f"\nOPTUNA CNN-TCN — {n_trials} trials | {CNN_D1_TCN_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
    config = WINDOW_CONFIGS["default"]
    session_windows = precompute_session_windows(df, config["window_size"], config["step_size"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, session_windows, num_classes, n_optuna_epochs),
        n_trials=n_trials, show_progress_bar=True,
        gc_after_trial=True, catch=(Exception,),
    )
    del session_windows; gc.collect()

    print(f"\nBest F1-Macro : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    OPTUNA_PATH_CNN_TCN.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_CNN_TCN, "w") as f:
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
    num_classes = int(model.output_shape[-1])
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

    optimizer = model.optimizer
    if optimizer is not None:
        optimizer = optimizer.__class__.from_config(optimizer.get_config())
    else:
        optimizer = Adam()
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

        models_dir = MODELS_DIR / "CNN-TCN"
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

        try:
            model_pruned = _apply_pruning_and_finetune(
                model, X_tr, y_tr, X_vl, y_vl, weight_dict,
                batch_size=best_params.get("batch_size", 32),
            )
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
            _free_memory(model_pruned)
        except Exception as exc:
            print(f"  Pruning échoué ({type(exc).__name__}: {exc})")
            traceback.print_exc()

        _save_edge_comparison_metrics(MODEL_NAME, edge_metrics)
        print(f"  Modèle global exporté (.tflite + .h).")
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
        return print(f"Dataset non trouvé : {DATA_MODEL_READY}")

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
        if USE_OPTUNA_CNN_TCN
        else MODEL_PARAMS["CNN_TCN"]
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
            f1_mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            plots_dir = BASE_DIR / "training_curves" / "CNN-TCN"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_fold(history, test_idx+1, test_part, test_sess,
                      plots_dir, f1_mac, bal_acc, y_test, y_pred)

            all_metrics.append({
                "Fold": test_idx+1, "Participant": int(test_part), "Session": int(test_sess),
                "F1_Macro": float(f1_mac), "Balanced_Accuracy": float(bal_acc),
            })

        except Exception as exc:
            print(f"  Fold {test_idx+1} échoué ({type(exc).__name__}: {exc})")
            traceback.print_exc()
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
    print(f"\n  Métriques → {METRICS_PATH}")

    train_global_model(df_labeled, df_unlabeled, best_params, num_classes)


if __name__ == "__main__":
    main()
