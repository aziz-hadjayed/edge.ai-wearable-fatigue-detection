"""
Student CNN micro — distillation de connaissances (Teacher ViT-1D).
Pipeline aligné sur train_cnn1.py : LOSO, Optuna, pseudo-labeling, export STM32.
"""
import gc
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.lite")
tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import (
    BatchNormalization, Conv1D, Dense, Dropout,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Input, MaxPooling1D,
)
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TerminateOnNaN
from models.semi_supervised import add_pseudo_labels
from models.Distillation.Teacher import (
    DISTILL_TEMP,
    MODEL_NAME as TEACHER_MODEL_NAME,
    PatchEmbedding,
    PositionalEncoding,
    TransformerBlock,
    build_teacher,
    generate_soft_targets,
    extract_all_windows,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════
MODEL_NAME = "DISTILL_STUDENT_CNN"
LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES = ["baseline", "activity", "pre_fatigue", "fatigue"]
N_OPTUNA_SESSIONS = 20
DEFAULT_ALPHA = 0.5

TEACHER_CUSTOM_OBJECTS = {
    "PatchEmbedding": PatchEmbedding,
    "PositionalEncoding": PositionalEncoding,
    "TransformerBlock": TransformerBlock,
}

STUDENT_OPTUNA_SPACE = {
    "n_conv_blocks": [1, 2, 3],
    "kernel_size": [3, 5],
    "filters": [8, 16, 32, 64],
    "use_batchnorm": [True, False],
    "activation": ["relu", "leaky_relu"],
    "pool_size": [2, 3],
    "global_pooling": ["avg", "max"],
    "l2_reg": (1e-5, 1e-2),
    "optimizer": ["adam", "rmsprop"],
    "dense_units": [32, 64, 128],
    "dropout_rate": (0.1, 0.5),
    "learning_rate": (1e-4, 1e-3),
    "batch_size": [16, 32, 64],
    "alpha": (0.3, 0.7),
}


def _free_memory(model=None):
    if model is not None:
        del model
    tf.keras.backend.clear_session()
    gc.collect()


# ══════════════════════════════════════════════════════════════════════
# EXPORT STM32
# ══════════════════════════════════════════════════════════════════════
def _export_c_array(tflite_bytes, h_path, stem):
    var_name = stem.replace("-", "_").replace(".", "_").lower()
    guard = var_name.upper() + "_H"
    hex_vals = [f"0x{b:02x}" for b in tflite_bytes]
    rows = ["  " + ", ".join(hex_vals[i : i + 16]) for i in range(0, len(hex_vals), 16)]
    lines = [
        "/* Auto-generated for STM32H7A3ZIT6Q — X-CUBE-AI */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f"const unsigned char {var_name}[] = {{",
        *[r + "," for r in rows],
        f"}};",
        f"const unsigned int {var_name}_len = {len(tflite_bytes)};",
        "",
        f"#endif /* {guard} */",
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
            yield [X_representative[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = _silent_tflite_convert(converter)

    tflite_path = models_dir / f"{stem}_int8.tflite"
    tflite_path.write_bytes(tflite_bytes)
    _export_c_array(tflite_bytes, models_dir / f"{stem}_int8.h", stem)
    print(f"  TFLite : {tflite_path.name}  ({len(tflite_bytes)/1024:.1f} KB / {STM32_FLASH_KB} KB)")


def build_inference_model(student_logits_model):
    """Modèle avec softmax pour inférence / export TFLite."""
    inputs = student_logits_model.input
    outputs = tf.keras.layers.Softmax(name="probs")(student_logits_model.output)
    inf_model = Model(inputs, outputs, name="Student_Inference")
    inf_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return inf_model


# ══════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════
def plot_fold(history, fold_idx, test_part, test_sess, save_dir,
              f1_mac, bal_acc, y_true=None, y_pred=None):
    import seaborn as sns
    epochs_range = range(1, len(history.history["loss"]) + 1)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    has_cm = (y_true is not None) and (y_pred is not None)
    ncols = 3 if has_cm else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    fig.suptitle(
        f"Student CNN  |  Fold {fold_idx} — P{test_part} S{test_sess}"
        f"  |  F1-Macro: {f1_mac:.3f}  |  Bal.Acc: {bal_acc:.3f}",
        fontsize=13, fontweight="bold",
    )

    axes[0].plot(epochs_range, history.history["loss"], label="Train Loss", color="#2196F3", linewidth=2)
    axes[0].plot(epochs_range, history.history["val_loss"], label="Val Loss", color="#F44336", linewidth=2, linestyle="--")
    axes[0].axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5, label=f"Best ({best_epoch})")
    axes[0].set_title("Loss — Train vs Validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
    axes[1].plot(epochs_range, history.history[acc_key], label="Train Acc", color="#4CAF50", linewidth=2)
    axes[1].plot(epochs_range, history.history[val_acc_key], label="Val Acc", color="#FF9800", linewidth=2, linestyle="--")
    axes[1].axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5, label=f"Best ({best_epoch})")
    axes[1].set_title("Accuracy — Train vs Validation")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    if has_cm:
        cm = confusion_matrix(y_true, y_pred)
        labels = [TARGET_NAMES[i] for i in np.unique(np.concatenate([y_true, y_pred]))]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
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
# MODÈLE STUDENT + LOSS DISTILLATION
# ══════════════════════════════════════════════════════════════════════
def make_distillation_loss(alpha, temperature, num_classes):
    """y_true = [hard OHE | soft probs teacher] concaténés sur l'axe features."""
    alpha = float(alpha)
    temperature = float(temperature)
    nc = int(num_classes)

    def loss(y_true, y_pred_logits):
        y_hard = y_true[:, :nc]
        y_soft = y_true[:, nc:]
        student_soft = tf.nn.softmax(y_pred_logits / temperature)
        kl = tf.keras.losses.KLDivergence()(y_soft, student_soft)
        ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_hard, y_pred_logits)
        return alpha * (temperature ** 2) * kl + (1.0 - alpha) * ce

    return loss


def build_student(trial, input_shape, num_classes):
    n_conv_blocks = trial.suggest_categorical("n_conv_blocks", STUDENT_OPTUNA_SPACE["n_conv_blocks"])
    kernel_size = trial.suggest_categorical("kernel_size", STUDENT_OPTUNA_SPACE["kernel_size"])
    filters = trial.suggest_categorical("filters", STUDENT_OPTUNA_SPACE["filters"])
    use_batchnorm = trial.suggest_categorical("use_batchnorm", STUDENT_OPTUNA_SPACE["use_batchnorm"])
    activation = trial.suggest_categorical("activation", STUDENT_OPTUNA_SPACE["activation"])
    pool_size = trial.suggest_categorical("pool_size", STUDENT_OPTUNA_SPACE["pool_size"])
    global_pooling = trial.suggest_categorical("global_pooling", STUDENT_OPTUNA_SPACE["global_pooling"])
    l2_reg = trial.suggest_float("l2_reg", *STUDENT_OPTUNA_SPACE["l2_reg"], log=True)
    optimizer_str = trial.suggest_categorical("optimizer", STUDENT_OPTUNA_SPACE["optimizer"])
    dense_units = trial.suggest_categorical("dense_units", STUDENT_OPTUNA_SPACE["dense_units"])
    dropout_rate = trial.suggest_float("dropout_rate", *STUDENT_OPTUNA_SPACE["dropout_rate"])
    learning_rate = trial.suggest_float("learning_rate", *STUDENT_OPTUNA_SPACE["learning_rate"], log=True)

    layers = [Input(shape=input_shape)]
    for i in range(n_conv_blocks):
        layers.append(Conv1D(filters, kernel_size, activation=activation, kernel_regularizer=l2(l2_reg)))
        if use_batchnorm:
            layers.append(BatchNormalization())
        if input_shape[0] // (pool_size ** (i + 1)) > 1:
            layers.append(MaxPooling1D(pool_size=pool_size))

    if global_pooling == "avg":
        layers.append(GlobalAveragePooling1D())
    else:
        layers.append(GlobalMaxPooling1D())

    layers += [
        Dense(dense_units, activation=activation, kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(num_classes, activation=None, name="logits"),
    ]
    model = Sequential(layers)
    opt = Adam(learning_rate) if optimizer_str == "adam" else RMSprop(learning_rate)
    return model, opt


def compile_student(model, opt, alpha, num_classes, temperature=DISTILL_TEMP):
    model.compile(
        optimizer=opt,
        loss=make_distillation_loss(alpha, temperature, num_classes),
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model


def make_distill_targets(y_hard, soft_probs, num_classes):
    return np.concatenate([to_categorical(y_hard, num_classes), soft_probs], axis=1).astype(np.float32)


def fold_stem_for_session(unique_sessions, part, sess):
    for i, (p, s) in enumerate(unique_sessions):
        if int(p) == int(part) and int(s) == int(sess):
            return f"fold{i+1:02d}_P{int(part)}_S{int(sess)}"
    raise FileNotFoundError(f"Session P{part} S{sess} absente des folds Teacher")


def sample_weights_from_labels(y, weight_dict):
    return np.array([weight_dict[int(lbl)] for lbl in y], dtype=np.float32)


def load_fold_teacher(fold_stem, window_size, n_features, num_classes):
    teacher_path = MODELS_DIR / "Teacher" / f"{fold_stem}_teacher.keras"
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher fold manquant : {teacher_path}")
    return tf.keras.models.load_model(teacher_path, custom_objects=TEACHER_CUSTOM_OBJECTS)


def get_soft_targets_for_X(teacher, X, temperature=DISTILL_TEMP, batch_size=64):
    _, soft = generate_soft_targets(teacher, X, temperature, batch_size)
    return soft


# ══════════════════════════════════════════════════════════════════════
# OPTUNA (Student uniquement)
# ══════════════════════════════════════════════════════════════════════
def optuna_objective(trial, df, num_classes, teacher_dir):
    config = WINDOW_CONFIGS["default"]
    window_size = config["window_size"]
    step_size = config["step_size"]
    batch_size = trial.suggest_categorical("batch_size", STUDENT_OPTUNA_SPACE["batch_size"])
    alpha = trial.suggest_float("alpha", *STUDENT_OPTUNA_SPACE["alpha"])

    import random
    unique_sessions = [
        tuple(x) for x in
        df[df[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
    ]
    val_sessions = random.sample(unique_sessions, min(N_OPTUNA_SESSIONS, len(unique_sessions)))
    scores = []

    for fold_idx, (val_part, val_sess) in enumerate(val_sessions):
        model = None
        teacher = None
        try:
            df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
            df_val = df[(df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess)].copy()

            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_val[SIGNAL_COLS] = scaler.transform(df_val[SIGNAL_COLS])

            X_train, y_train = extract_all_windows(df_train, window_size, step_size)
            X_val, y_val = extract_all_windows(df_val, window_size, step_size)
            if len(X_val) == 0:
                continue

            fold_stem = fold_stem_for_session(unique_sessions, val_part, val_sess)
            try:
                teacher = load_fold_teacher(fold_stem, window_size, len(SIGNAL_COLS), num_classes)
            except FileNotFoundError:
                teacher = None
            if teacher is None:
                continue

            soft_train = get_soft_targets_for_X(teacher, X_train, batch_size=batch_size)
            y_distill = make_distill_targets(y_train, soft_train, num_classes)

            w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), w))

            model, opt = build_student(trial, (window_size, len(SIGNAL_COLS)), num_classes)
            compile_student(model, opt, alpha, num_classes)
            sw = sample_weights_from_labels(y_train, class_weight_dict)
            model.fit(
                X_train, y_distill,
                epochs=10, batch_size=batch_size,
                sample_weight=sw,
                callbacks=[TerminateOnNaN()], verbose=0,
            )
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            score = f1_score(y_val, y_pred, average="macro", zero_division=0)
            scores.append(score)

            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        except tf.errors.ResourceExhaustedError:
            raise optuna.exceptions.TrialPruned("OOM")
        except Exception as exc:
            print(f"  [WARN] fold Optuna échoué ({type(exc).__name__}: {exc})")
            scores.append(0.0)
        finally:
            _free_memory(model)
            _free_memory(teacher)

    return float(np.mean(scores)) if scores else 0.0


def optimize_hyperparams(df, num_classes, n_trials=50):
    print(f"\nOPTUNA Student CNN — {n_trials} trials | {N_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, df, num_classes, MODELS_DIR / "Teacher"),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
        catch=(Exception,),
    )
    print(f"\nBest F1-Macro : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    OPTUNA_PATH_DISTILL_STUDENT.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_DISTILL_STUDENT, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return study.best_params


# ══════════════════════════════════════════════════════════════════════
# MODÈLE GLOBAL
# ══════════════════════════════════════════════════════════════════════
def train_global_model(df_labeled, df_unlabeled, best_params, num_classes):
    print("\n" + "=" * 60 + f"\nMODÈLE GLOBAL — {MODEL_NAME}\n" + "=" * 60)
    config = WINDOW_CONFIGS["default"]
    W_SIZE, S_SIZE, EPOCHS = config["window_size"], config["step_size"], config["epochs"]
    alpha = best_params.get("alpha", DEFAULT_ALPHA)
    batch_size = best_params.get("batch_size", 32)

    teacher_path = MODELS_DIR / "Teacher" / f"{TEACHER_MODEL_NAME}_final.keras"
    if not teacher_path.exists():
        print(f"  ❌ Teacher final introuvable : {teacher_path}")
        return

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

    teacher = tf.keras.models.load_model(teacher_path, custom_objects=TEACHER_CUSTOM_OBJECTS)
    soft_all = get_soft_targets_for_X(teacher, X_all, batch_size=batch_size)
    y_distill = make_distill_targets(y_all, soft_all, num_classes)
    _free_memory(teacher)

    model = None
    try:
        idx = np.random.default_rng(42).permutation(len(X_all))
        split = int(0.9 * len(X_all))
        X_tr = X_all[idx[:split]]
        y_tr_hard = y_all[idx[:split]]
        y_tr = y_distill[idx[:split]]
        X_vl = X_all[idx[split:]]
        y_vl = y_distill[idx[split:]]

        model, opt = build_student(optuna.trial.FixedTrial(best_params), (W_SIZE, len(SIGNAL_COLS)), num_classes)
        compile_student(model, opt, alpha, num_classes)
        model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),
            epochs=EPOCHS,
            batch_size=batch_size,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                TerminateOnNaN(),
            ],
            verbose=1,
        )

        inf_model = build_inference_model(model)
        X_tr_v2, y_tr_v2, pseudo_y, _ = add_pseudo_labels(inf_model, X_tr, y_tr_hard, X_unlabeled)
        if len(pseudo_y) > 0:
            teacher = tf.keras.models.load_model(teacher_path, custom_objects=TEACHER_CUSTOM_OBJECTS)
            soft_tr_v2 = get_soft_targets_for_X(teacher, X_tr_v2, batch_size=batch_size)
            y_tr_v2_dist = make_distill_targets(y_tr_v2, soft_tr_v2, num_classes)
            _free_memory(teacher, inf_model)
            _free_memory(model)
            model, opt = build_student(optuna.trial.FixedTrial(best_params), (W_SIZE, len(SIGNAL_COLS)), num_classes)
            compile_student(model, opt, alpha, num_classes)
            model.fit(
                X_tr_v2, y_tr_v2_dist,
                validation_data=(X_vl, y_vl),
                epochs=EPOCHS,
                batch_size=batch_size,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                    TerminateOnNaN(),
                ],
                verbose=1,
            )
            inf_model = build_inference_model(model)
            X_repr = np.concatenate([X_tr_v2, X_vl], axis=0)
        else:
            X_repr = X_all

        models_dir = MODELS_DIR / "Distillation"
        models_dir.mkdir(parents=True, exist_ok=True)
        _save_tflite_int8(inf_model, X_repr, models_dir, f"{MODEL_NAME}_global")
        print("  ✅ Modèle global Student exporté (.tflite + .h).")
    except Exception as exc:
        print(f"  ❌ Modèle global échoué ({type(exc).__name__}: {exc})")
    finally:
        _free_memory(model)
        del X_all, y_all, y_distill
        gc.collect()


# ══════════════════════════════════════════════════════════════════════
# BOUCLE LOSO
# ══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60 + f"\nTRAIN LOSO — {MODEL_NAME} (Distillation)\n" + "=" * 60)
    if not DATA_MODEL_READY.exists():
        return print(f"❌ Dataset non trouvé : {DATA_MODEL_READY}")

    teacher_dir = MODELS_DIR / "Teacher"
    if not teacher_dir.exists() or not list(teacher_dir.glob("*_teacher.keras")):
        return print(
            f"❌ Teachers LOSO introuvables dans {teacher_dir}. "
            "Lancez d'abord : python -m models.Distillation.Teacher"
        )

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
        if USE_OPTUNA_DISTILL_STUDENT
        else MODEL_PARAMS["DISTILL_STUDENT"]
    )
    alpha = best_params.get("alpha", DEFAULT_ALPHA)

    all_metrics = []
    import random
    random.seed(42)

    plots_dir = BASE_DIR / "training_curves" / "Distillation"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for test_idx, (test_part, test_sess) in enumerate(unique_sessions):
        print(f"\n--- FOLD {test_idx+1}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")
        config = WINDOW_CONFIGS.get(test_part, WINDOW_CONFIGS["default"])
        W_SIZE, S_SIZE, EPOCHS = config["window_size"], config["step_size"], config["epochs"]

        df_pool = df_labeled[~((df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess))].copy()
        df_test = df_labeled[(df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess)].copy()

        pool_sessions = [tuple(x) for x in df_pool[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
        val_part, val_sess = random.choice(pool_sessions)
        print(f"  Val : P{val_part} S{val_sess}")

        df_fit = df_pool[~((df_pool[COL_PARTICIPANT] == val_part) & (df_pool[COL_SESSION] == val_sess))].copy()
        df_val = df_pool[(df_pool[COL_PARTICIPANT] == val_part) & (df_pool[COL_SESSION] == val_sess)].copy()

        scaler = RobustScaler()
        df_fit[SIGNAL_COLS] = scaler.fit_transform(df_fit[SIGNAL_COLS])
        df_val[SIGNAL_COLS] = scaler.transform(df_val[SIGNAL_COLS])
        df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])
        df_unlabeled_fit = df_unlabeled.copy()
        if len(df_unlabeled_fit) > 0:
            df_unlabeled_fit[SIGNAL_COLS] = scaler.transform(df_unlabeled_fit[SIGNAL_COLS])

        X_fit, y_fit = extract_all_windows(df_fit, W_SIZE, S_SIZE)
        X_val, y_val = extract_all_windows(df_val, W_SIZE, S_SIZE)
        X_test, y_test = extract_all_windows(df_test, W_SIZE, S_SIZE)
        X_unlabeled, _ = (
            extract_all_windows(df_unlabeled_fit, W_SIZE, S_SIZE)
            if len(df_unlabeled_fit) > 0 else (np.array([]), np.array([]))
        )

        if len(X_test) == 0:
            print(f"  WARNING: test vide pour P{test_part}. Skip.")
            continue
        if len(X_val) == 0:
            print(f"  WARNING: val vide pour P{val_part}. Skip.")
            continue

        fold_stem = f"fold{test_idx+1:02d}_P{test_part}_S{test_sess}"
        try:
            teacher = load_fold_teacher(fold_stem, W_SIZE, len(SIGNAL_COLS), num_classes)
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
            continue

        print(f"  Fit: {len(X_fit)} | Val: {len(X_val)} | Test: {len(X_test)} fenêtres")
        soft_fit = get_soft_targets_for_X(teacher, X_fit, batch_size=best_params.get("batch_size", 32))
        y_distill_fit = make_distill_targets(y_fit, soft_fit, num_classes)
        y_distill_val = make_distill_targets(y_val, get_soft_targets_for_X(teacher, X_val), num_classes)
        _free_memory(teacher)

        w = compute_class_weight("balanced", classes=np.unique(y_fit), y=y_fit)
        weight_dict = dict(zip(np.unique(y_fit), w))
        sw_fit = sample_weights_from_labels(y_fit, weight_dict)

        model = None
        try:
            model, opt = build_student(optuna.trial.FixedTrial(best_params), (W_SIZE, len(SIGNAL_COLS)), num_classes)
            compile_student(model, opt, alpha, num_classes)
            history = model.fit(
                X_fit, y_distill_fit,
                epochs=EPOCHS,
                validation_data=(X_val, y_distill_val),
                batch_size=best_params.get("batch_size", 32),
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                    TerminateOnNaN(),
                ],
                sample_weight=sw_fit,
                verbose=1,
            )

            inf_model = build_inference_model(model)
            X_fit_v2, y_fit_v2, pseudo_y, _ = add_pseudo_labels(inf_model, X_fit, y_fit, X_unlabeled)
            if len(pseudo_y) > 0:
                teacher = load_fold_teacher(fold_stem, W_SIZE, len(SIGNAL_COLS), num_classes)
                soft_fit_v2 = get_soft_targets_for_X(teacher, X_fit_v2)
                _free_memory(teacher)
                y_distill_v2 = make_distill_targets(y_fit_v2, soft_fit_v2, num_classes)
                w_v2 = compute_class_weight("balanced", classes=np.unique(y_fit_v2), y=y_fit_v2)
                weight_dict_v2 = dict(zip(np.unique(y_fit_v2), w_v2))
                sw_v2 = sample_weights_from_labels(y_fit_v2, weight_dict_v2)
                _free_memory(model)
                model, opt = build_student(optuna.trial.FixedTrial(best_params), (W_SIZE, len(SIGNAL_COLS)), num_classes)
                compile_student(model, opt, alpha, num_classes)
                history = model.fit(
                    X_fit_v2, y_distill_v2,
                    epochs=EPOCHS,
                    validation_data=(X_val, y_distill_val),
                    batch_size=best_params.get("batch_size", 32),
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                        TerminateOnNaN(),
                    ],
                    sample_weight=sw_v2,
                    verbose=1,
                )
                inf_model = build_inference_model(model)

            y_pred = np.argmax(inf_model.predict(X_test, verbose=0), axis=1)
            f1_mac = f1_score(y_test, y_pred, average="macro")
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  ✅ F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            plot_fold(history, test_idx + 1, test_part, test_sess, plots_dir, f1_mac, bal_acc, y_test, y_pred)

            all_metrics.append({
                "Fold": test_idx + 1,
                "Participant": int(test_part),
                "Session": int(test_sess),
                "F1_Macro": float(f1_mac),
                "Balanced_Accuracy": float(bal_acc),
            })

        except Exception as exc:
            print(f"  ❌ Fold {test_idx+1} échoué ({type(exc).__name__}: {exc})")
        finally:
            _free_memory(model)
            del X_fit, X_val, X_test, y_fit, y_val, y_test
            gc.collect()

    if not all_metrics:
        return print("Aucun fold Student complété.")

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
            curr_metrics = {}
    curr_metrics[MODEL_NAME] = {
        "mean_f1": float(df_res["F1_Macro"].mean()),
        "std_f1": float(df_res["F1_Macro"].std()),
        "mean_bal_acc": float(df_res["Balanced_Accuracy"].mean()),
        "std_bal_acc": float(df_res["Balanced_Accuracy"].std()),
        "params": best_params,
        "distill_temperature": DISTILL_TEMP,
        "folds": all_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(curr_metrics, f, indent=4)
    print(f"\n📊 Métriques → {METRICS_PATH}")

    train_global_model(df_labeled, df_unlabeled, best_params, num_classes)


if __name__ == "__main__":
    main()
