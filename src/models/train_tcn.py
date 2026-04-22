import gc
import json
import os
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Conv1D,
    Dense, Dropout, GlobalAveragePooling1D, Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ══════════════════════════════════════════════════════════════════════
# 2. CONSTANTES
# ══════════════════════════════════════════════════════════════════════
MODEL_NAME        = "TCN_LOSO"
LABEL_MAPPING     = {-1: 0, 0: 1, 1: 2}
TARGET_NAMES      = ["baseline (-1)", "activity (0)", "fatigue (1)"]
USE_SMOTE         = True    # rééquilibrage SMOTE sur X_fit uniquement
N_OPTUNA_SESSIONS = 11      # sessions tirées au hasard pour évaluer chaque trial Optuna

# Budget Flash STM32 pour TFLite INT8
EDGE_FLASH_BUDGET_KB = STM32_FLASH_KB

OPTUNA_SPACE = {
    "n_tcn_blocks":  [2, 3, 4, 5],
    "tcn_filters":   [32, 64, 128],
    "tcn_kernel":    [3, 5, 7],
    "activation":    ["relu", "leaky_relu"],
    "l2_reg":        (1e-5, 1e-2),
    "optimizer":     ["adam", "rmsprop"],
    "dense_units":   [32, 64, 128],
    "dropout_rate":  (0.2, 0.5),
    "learning_rate": (1e-4, 1e-3),
    "batch_size":    [32, 64, 128],
}

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
    tflite_bytes = converter.convert()

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
# 6. DONNÉES — fenêtrage + SMOTE + pré-calcul Optuna
# ══════════════════════════════════════════════════════════════════════
def extract_windows(group_df, window_size, step_size):
    X = group_df[SIGNAL_COLS].values
    y = group_df[COL_LABEL].values
    wx, wy = [], []
    for start in range(0, len(X) - window_size + 1, step_size):
        end = start + window_size
        vals, counts = np.unique(y[start:end], return_counts=True)
        wx.append(X[start:end])
        wy.append(vals[np.argmax(counts)])
    return wx, wy

def extract_all_windows(df, window_size, step_size):
    X_all, y_all = [], []
    if COL_TIMESTAMP in df.columns:
        df = df.sort_values([COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP])
    for _, group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        wx, wy = extract_windows(group, window_size, step_size)
        X_all.extend(wx); y_all.extend(wy)
    return np.array(X_all), np.array(y_all)

def apply_smote(X, y, label="fit"):
    n_samples, timesteps, n_features = X.shape
    unique, counts = np.unique(y, return_counts=True)
    k = min(5, counts.min() - 1)
    if k < 1:
        print(f"  SMOTE [{label}] ignoré — classe trop petite ({counts.min()} samples)")
        return X, y
    print(f"  SMOTE [{label}] avant : { {int(u): int(c) for u, c in zip(unique, counts)} }")
    X_res, y_res = SMOTE(k_neighbors=k, random_state=42).fit_resample(
        X.reshape(n_samples, -1), y
    )
    unique2, counts2 = np.unique(y_res, return_counts=True)
    print(f"  SMOTE [{label}] après : { {int(u): int(c) for u, c in zip(unique2, counts2)} }")
    return X_res.reshape(-1, timesteps, n_features), y_res

def precompute_session_windows(df, window_size, step_size):
    """Pré-extrait les fenêtres brutes (float16) par session — une seule fois pour Optuna."""
    session_windows = {}
    if COL_TIMESTAMP in df.columns:
        df = df.sort_values([COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP])
    for (part, sess), group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        wx, wy = extract_windows(group, window_size, step_size)
        if wx:
            session_windows[(part, sess)] = (np.array(wx, dtype=np.float16), np.array(wy))
    total_mb = sum(x.nbytes for x, _ in session_windows.values()) / 1024 ** 2
    print(f"  Fenêtres pré-calculées : {len(session_windows)} sessions — {total_mb:.1f} MB RAM")
    return session_windows

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
    n_tcn_blocks  = trial.suggest_categorical("n_tcn_blocks", OPTUNA_SPACE["n_tcn_blocks"])
    tcn_filters   = trial.suggest_categorical("tcn_filters",  OPTUNA_SPACE["tcn_filters"])
    tcn_kernel    = trial.suggest_categorical("tcn_kernel",   OPTUNA_SPACE["tcn_kernel"])
    activation    = trial.suggest_categorical("activation",   OPTUNA_SPACE["activation"])
    l2_reg        = trial.suggest_float("l2_reg",       *OPTUNA_SPACE["l2_reg"], log=True)
    optimizer_str = trial.suggest_categorical("optimizer",    OPTUNA_SPACE["optimizer"])
    dense_units   = trial.suggest_categorical("dense_units",  OPTUNA_SPACE["dense_units"])
    dropout_rate  = trial.suggest_float("dropout_rate",  *OPTUNA_SPACE["dropout_rate"])
    learning_rate = trial.suggest_float("learning_rate", *OPTUNA_SPACE["learning_rate"], log=True)

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
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=["accuracy"], jit_compile=False)
    return model

def optuna_objective(trial, session_windows, num_classes, n_optuna_epochs=5):
    config      = WINDOW_CONFIGS["default"]
    window_size = config["window_size"]
    input_shape = (window_size, len(SIGNAL_COLS))
    batch_size  = trial.suggest_categorical("batch_size", OPTUNA_SPACE["batch_size"])

    import random
    all_sessions  = list(session_windows.keys())
    eval_sessions = random.sample(all_sessions, min(N_OPTUNA_SESSIONS, len(all_sessions)))
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
            del X_train, X_val, y_train, y_val

    return float(np.mean(scores)) if scores else 0.0

def optimize_hyperparams(df, num_classes, n_trials=50, n_optuna_epochs=10):
    print(f"\nOPTUNA TCN — {n_trials} trials | {N_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
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
    OPTUNA_PATH_TCN.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_TCN, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return study.best_params

# ══════════════════════════════════════════════════════════════════════
# 8. BOUCLE LOSO
# ══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60 + f"\nTRAIN LOSO — {MODEL_NAME}\n" + "=" * 60)
    if not DATA_MODEL_READY.exists():
        return print(f"❌ Dataset non trouvé : {DATA_MODEL_READY}")

    df = pd.read_csv(DATA_MODEL_READY)
    df[COL_LABEL] = df[COL_LABEL].map(LABEL_MAPPING)

    unique_sessions = [
        tuple(x) for x in
        df[df[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
    ]
    num_classes = len(LABEL_MAPPING)
    print(f"Sessions détectées : {len(unique_sessions)}")

    best_params = (
        optimize_hyperparams(df, num_classes)
        if USE_OPTUNA_TCN
        else MODEL_PARAMS["TCN"]
    )

    all_metrics = []
    import random
    random.seed(42)

    for test_idx, (test_part, test_sess) in enumerate(unique_sessions):
        print(f"\n--- FOLD {test_idx+1}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")
        config = WINDOW_CONFIGS.get(test_part, WINDOW_CONFIGS["default"])
        W_SIZE, S_SIZE, EPOCHS = config["window_size"], config["step_size"], config["epochs"]

        df_pool = df[~((df[COL_PARTICIPANT] == test_part) & (df[COL_SESSION] == test_sess))].copy()
        df_test = df[ (df[COL_PARTICIPANT] == test_part)  & (df[COL_SESSION] == test_sess)].copy()

        pool_sessions      = [tuple(x) for x in df_pool[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
        val_part, val_sess = random.choice(pool_sessions)
        print(f"  Val : P{val_part} S{val_sess}")

        df_fit = df_pool[~((df_pool[COL_PARTICIPANT] == val_part) & (df_pool[COL_SESSION] == val_sess))].copy()
        df_val = df_pool[ (df_pool[COL_PARTICIPANT] == val_part)  & (df_pool[COL_SESSION] == val_sess)].copy()

        scaler = RobustScaler()
        df_fit[SIGNAL_COLS]  = scaler.fit_transform(df_fit[SIGNAL_COLS])
        df_val[SIGNAL_COLS]  = scaler.transform(df_val[SIGNAL_COLS])
        df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])

        X_fit,  y_fit  = extract_all_windows(df_fit,  W_SIZE, S_SIZE)
        X_val,  y_val  = extract_all_windows(df_val,  W_SIZE, S_SIZE)
        X_test, y_test = extract_all_windows(df_test, W_SIZE, S_SIZE)

        if len(X_test) == 0:
            print(f"  WARNING: test vide pour P{test_part}. Skip."); continue
        if len(X_val) == 0:
            print(f"  WARNING: val vide pour P{val_part}. Skip.");   continue

        print(f"  Fit: {len(X_fit)} | Val: {len(X_val)} | Test: {len(X_test)} fenêtres")

        if USE_SMOTE:
            X_fit, y_fit = apply_smote(X_fit, y_fit, label=f"fold{test_idx+1}")
            weight_dict  = None
        else:
            w           = compute_class_weight("balanced", classes=np.unique(y_fit), y=y_fit)
            weight_dict = dict(enumerate(w))

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

            y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
            f1_mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  ✅ F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            plots_dir = BASE_DIR / "training_curves" / "TCN"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_fold(history, test_idx+1, test_part, test_sess,
                      plots_dir, f1_mac, bal_acc, y_test, y_pred)

            models_dir = MODELS_DIR / "TCN"
            models_dir.mkdir(parents=True, exist_ok=True)
            stem = f"TCN_fold{test_idx+1}_testP{test_part}_S{test_sess}"
            _save_tflite_int8(model, X_fit, models_dir, stem)

            all_metrics.append({
                "Fold": test_idx+1, "Participant": int(test_part), "Session": int(test_sess),
                "F1_Macro": float(f1_mac), "Balanced_Accuracy": float(bal_acc),
            })

        except Exception as exc:
            print(f"  ❌ Fold {test_idx+1} échoué ({type(exc).__name__}: {exc})")
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
    print(f"\n📊 Métriques → {METRICS_PATH}")


if __name__ == "__main__":
    main()
