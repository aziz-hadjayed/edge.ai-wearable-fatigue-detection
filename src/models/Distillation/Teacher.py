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

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

from keras.layers import (
    Dense, Dropout, GlobalAveragePooling1D, Input, Layer, LayerNormalization,
    MultiHeadAttention, Reshape, Add, Softmax,
)
from models.semi_supervised import add_pseudo_labels
from keras.models import Model, Sequential
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras.utils import to_categorical

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ══════════════════════════════════════════════════════════════════════
# 2. CONSTANTES
# ══════════════════════════════════════════════════════════════════════
MODEL_NAME        = "TRANSFORMER_TEACHER"
LABEL_MAPPING     = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES      = ["baseline", "activity", "pre_fatigue", "fatigue"]

# Espace Optuna Teacher (patch_size doit diviser window_size, ex. 240)
TEACHER_OPTUNA_SPACE = {
    "patch_size": [8, 12, 16, 20, 24, 30, 40],
    "d_model": [32, 64, 96, 128],
    "num_heads": [2, 4, 8],
    "num_layers": [2, 3, 4, 5],
    "d_ff": [128, 256, 512],
    "dropout": (0.05, 0.35),
    "dense_units": [32, 64, 128],
    "learning_rate": (1e-5, 5e-4),
    "weight_decay": (1e-5, 1e-3),
    "batch_size": [32, 64, 128],
}

# Température pour distillation
DISTILL_TEMP = 4.0


def teacher_cfg_from_params(params, epochs):
    """Construit la config d'entraînement Teacher à partir des hyperparamètres Optuna."""
    return {
        "patch_size": int(params["patch_size"]),
        "d_model": int(params["d_model"]),
        "num_heads": int(params["num_heads"]),
        "d_ff": int(params["d_ff"]),
        "num_layers": int(params["num_layers"]),
        "dropout": float(params["dropout"]),
        "dense_units": int(params["dense_units"]),
        "learning_rate": float(params["learning_rate"]),
        "weight_decay": float(params.get("weight_decay", 1e-4)),
        "batch_size": int(params.get("batch_size", 64)),
        "epochs": int(epochs),
    }


def validate_teacher_cfg(cfg, window_size):
    if window_size % cfg["patch_size"] != 0:
        raise ValueError(
            f"window_size ({window_size}) non divisible par patch_size ({cfg['patch_size']})"
        )
    if cfg["d_model"] % cfg["num_heads"] != 0:
        raise ValueError(
            f"d_model ({cfg['d_model']}) doit être divisible par num_heads ({cfg['num_heads']})"
        )


def suggest_teacher_cfg(trial, window_size):
    patch_size = trial.suggest_categorical("patch_size", TEACHER_OPTUNA_SPACE["patch_size"])
    if window_size % patch_size != 0:
        raise optuna.exceptions.TrialPruned(f"patch_size={patch_size} incompatible avec W={window_size}")

    d_model = trial.suggest_categorical("d_model", TEACHER_OPTUNA_SPACE["d_model"])
    num_heads = trial.suggest_categorical("num_heads", TEACHER_OPTUNA_SPACE["num_heads"])
    if d_model % num_heads != 0:
        raise optuna.exceptions.TrialPruned(f"d_model={d_model}, num_heads={num_heads}")

    return {
        "patch_size": patch_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": trial.suggest_categorical("num_layers", TEACHER_OPTUNA_SPACE["num_layers"]),
        "d_ff": trial.suggest_categorical("d_ff", TEACHER_OPTUNA_SPACE["d_ff"]),
        "dropout": trial.suggest_float("dropout", *TEACHER_OPTUNA_SPACE["dropout"]),
        "dense_units": trial.suggest_categorical("dense_units", TEACHER_OPTUNA_SPACE["dense_units"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *TEACHER_OPTUNA_SPACE["learning_rate"], log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", *TEACHER_OPTUNA_SPACE["weight_decay"], log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", TEACHER_OPTUNA_SPACE["batch_size"]),
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
# 4. MODULE TRANSFORMER — PATCH EMBEDDING
# ══════════════════════════════════════════════════════════════════════
class PatchEmbedding(Layer):
    """
    Conv1D agit comme projection linéaire par patch.
    Input:  (batch, W, C)  →  Output: (batch, N, D)
    N = W // patch_size, D = d_model
    """
    def __init__(self, patch_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.d_model = d_model
        self.proj = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            name='patch_embed'
        )

    def call(self, x):
        # x: (batch, W, C)
        x = self.proj(x)  # (batch, N, D)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "d_model": self.d_model})
        return config

# ══════════════════════════════════════════════════════════════════════
# 5. MODULE TRANSFORMER — POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════
class PositionalEncoding(Layer):
    """
    Encodage positionnel sinusoïdal fixe.
    Input:  (batch, N, D)  →  Output: (batch, N, D)
    """
    def __init__(self, max_seq_len=1024, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len

    def build(self, input_shape):
        d_model = int(input_shape[-1])
        positions = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((self.max_seq_len, d_model))
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_seq_len": self.max_seq_len})
        return config

# ══════════════════════════════════════════════════════════════════════
# 6. MODULE TRANSFORMER — BLOC TRANSFORMER
# ══════════════════════════════════════════════════════════════════════
class TransformerBlock(Layer):
    """
    Bloc standard: Multi-Head Self-Attention + FFN
    Input:  (batch, N, D)  →  Output: (batch, N, D)
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name='mha'
        )
        self.ln1 = LayerNormalization(epsilon=1e-6, name='ln1')
        self.ln2 = LayerNormalization(epsilon=1e-6, name='ln2')

        self.ffn = Sequential([
            Dense(d_ff, activation='gelu', name='ffn_dense1'),
            Dropout(dropout_rate),
            Dense(d_model, name='ffn_dense2'),
            Dropout(dropout_rate),
        ], name='ffn')

        self.add1 = Add(name='residual1')
        self.add2 = Add(name='residual2')

    def call(self, x, training=None):
        # Self-attention avec residual
        attn_out = self.attn(x, x, training=training)
        x = self.add1([x, attn_out])
        x = self.ln1(x)

        # FFN avec residual
        ffn_out = self.ffn(x, training=training)
        x = self.add2([x, ffn_out])
        x = self.ln2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
        })
        return config

# ══════════════════════════════════════════════════════════════════════
# 7. CONSTRUCTION DU TEACHER
# ══════════════════════════════════════════════════════════════════════
def build_teacher(window_size, n_features, num_classes, cfg):
    """
    Data Flow:
    (batch, W, C) → PatchEmbed → (batch, N, D) → PosEnc → [Transformer×L] 
    → GAP → (batch, D) → Dense → Dropout → (batch, num_classes) → logits
    """
    inputs = Input(shape=(window_size, n_features), name='input')

    # Patch Embedding: (W, C) → (N, D)
    x = PatchEmbedding(cfg["patch_size"], cfg["d_model"], name='patch_embed')(inputs)

    # Vérification dimensionnelle
    n_patches = window_size // cfg["patch_size"]
    x = Reshape((n_patches, cfg["d_model"]), name='reshape_patches')(x)

    # Positional Encoding
    x = PositionalEncoding(max_seq_len=1024, name='pos_enc')(x)

    # Transformer Blocks × L
    for i in range(cfg["num_layers"]):
        x = TransformerBlock(
            cfg["d_model"], cfg["num_heads"], cfg["d_ff"],
            cfg["dropout"], name=f'transformer_{i+1}'
        )(x)

    # Global Average Pooling: (batch, N, D) → (batch, D)
    x = GlobalAveragePooling1D(name='gap')(x)

    # Tête de classification
    x = Dense(cfg["dense_units"], activation='gelu', name='dense_head')(x)
    x = Dropout(cfg["dropout"], name='dropout_head')(x)
    outputs = Dense(num_classes, name='logits')(x)  # Pas de softmax ici

    model = Model(inputs, outputs, name='Teacher_Transformer')

    model.compile(
        optimizer=AdamW(
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg.get("weight_decay", 1e-4),
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# 8. OPTUNA — Teacher
# ══════════════════════════════════════════════════════════════════════
def optuna_objective(trial, df, num_classes):
    config = WINDOW_CONFIGS["default"]
    window_size = config["window_size"]
    step_size = config["step_size"]

    try:
        params = suggest_teacher_cfg(trial, window_size)
    except optuna.exceptions.TrialPruned:
        raise

    cfg = teacher_cfg_from_params(params, epochs=10)
    validate_teacher_cfg(cfg, window_size)

    import random
    unique_sessions = [
        tuple(x) for x in
        df[df[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
    ]
    val_sessions = random.sample(
        unique_sessions, min(TEACHER_OPTUNA_SESSIONS, len(unique_sessions))
    )
    scores = []

    for fold_idx, (val_part, val_sess) in enumerate(val_sessions):
        model = None
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

            w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
            weight_dict = dict(zip(np.unique(y_train), w))

            model = build_teacher(window_size, len(SIGNAL_COLS), num_classes, cfg)
            model.fit(
                X_train, to_categorical(y_train, num_classes),
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                validation_split=0.1,
                class_weight=weight_dict,
                callbacks=[TerminateOnNaN()],
                verbose=0,
            )
            y_pred = _predict_classes(model, X_val, cfg["batch_size"])
            score = f1_score(y_val, y_pred, average="macro", zero_division=0)
            scores.append(score)

            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        except tf.errors.ResourceExhaustedError:
            raise optuna.exceptions.TrialPruned("OOM")
        except Exception as exc:
            print(f"  [WARN] fold Optuna Teacher ({type(exc).__name__}: {exc})")
            scores.append(0.0)
        finally:
            _free_memory(model)
            for var in ("X_train", "X_val", "y_train", "y_val"):
                if var in locals():
                    del locals()[var]

    return float(np.mean(scores)) if scores else 0.0


def optimize_hyperparams(df, num_classes, n_trials=None):
    n_trials = n_trials or TEACHER_OPTUNA_TRIALS
    print(
        f"\nOPTUNA Teacher ViT-1D — {n_trials} trials | {TEACHER_OPTUNA_SESSIONS} sessions\n"
        + "=" * 60
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, df, num_classes),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
        catch=(Exception,),
    )
    print(f"\nBest F1-Macro : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    OPTUNA_PATH_DISTILL_TEACHER.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_DISTILL_TEACHER, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return study.best_params


# ══════════════════════════════════════════════════════════════════════
# 9. FONCTION DE DISTILLATION (soft targets)
# ══════════════════════════════════════════════════════════════════════
def generate_soft_targets(teacher_model, X, temperature=DISTILL_TEMP, batch_size=64):
    """
    Génère les logits et probabilités tempérées du teacher.
    Retourne: (logits, soft_probs)
    soft_probs = softmax(logits / T)
    """
    logits = teacher_model.predict(X, batch_size=batch_size, verbose=0)
    # Température
    logits_temp = logits / temperature
    # Softmax stable
    soft_probs = tf.nn.softmax(logits_temp).numpy()
    return logits, soft_probs

# ══════════════════════════════════════════════════════════════════════
# 9. SAUVEGARDE DES SOFT TARGETS
# ══════════════════════════════════════════════════════════════════════
def save_soft_targets(soft_probs, logits, y_true, save_path):
    """Sauvegarde au format NPZ pour chargement rapide par le Student."""
    np.savez_compressed(
        save_path,
        soft_probs=soft_probs.astype(np.float32),
        logits=logits.astype(np.float32),
        y_true=y_true.astype(np.int32),
    )
    size_mb = (soft_probs.nbytes + logits.nbytes) / 1e6
    print(f"  Soft targets sauvegardés : {save_path} ({size_mb:.1f} MB)")


def _predict_classes(model, X, batch_size=64):
    logits = model.predict(X, batch_size=batch_size, verbose=0)
    return np.argmax(logits, axis=1)


def build_teacher_inference_model(teacher_logits_model):
    """Modèle Teacher avec softmax pour pseudo-labeling et inférence."""
    inputs = teacher_logits_model.input
    outputs = Softmax(name="probs")(teacher_logits_model.output)
    inf_model = Model(inputs, outputs, name="Teacher_Inference")
    inf_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return inf_model


def plot_fold(history, fold_idx, test_part, test_sess, save_dir,
              f1_mac, bal_acc, y_true=None, y_pred=None):
    import seaborn as sns
    epochs_range = range(1, len(history.history["loss"]) + 1)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    has_cm = (y_true is not None) and (y_pred is not None)
    ncols = 3 if has_cm else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    fig.suptitle(
        f"Teacher ViT-1D  |  Fold {fold_idx} — P{test_part} S{test_sess}"
        f"  |  F1-Macro: {f1_mac:.3f}  |  Bal.Acc: {bal_acc:.3f}",
        fontsize=13, fontweight="bold",
    )

    axes[0].plot(epochs_range, history.history["loss"], label="Train Loss", color="#2196F3", linewidth=2)
    axes[0].plot(epochs_range, history.history["val_loss"], label="Val Loss", color="#F44336", linewidth=2, linestyle="--")
    axes[0].axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5, label=f"Best ({best_epoch})")
    axes[0].set_title("Loss — Train vs Validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history.history["accuracy"], label="Train Acc", color="#4CAF50", linewidth=2)
    axes[1].plot(epochs_range, history.history["val_accuracy"], label="Val Acc", color="#FF9800", linewidth=2, linestyle="--")
    axes[1].axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5, label=f"Best ({best_epoch})")
    axes[1].set_title("Accuracy — Train vs Validation")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    if has_cm:
        cm = confusion_matrix(y_true, y_pred)
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
# 10. ENTRAÎNEMENT DU TEACHER (LOSO)
# ══════════════════════════════════════════════════════════════════════
def train_teacher_loso(df_labeled, num_classes, teacher_params, df_unlabeled=None):
    """
    Entraîne le Teacher en LOSO (+ pseudo-labeling semi-supervisé sur le pool d'entraînement).
    Pour chaque fold : modèle + soft targets. Puis teacher final sur toutes les données labellisées.
    """
    if df_unlabeled is None:
        df_unlabeled = pd.DataFrame()
    else:
        df_unlabeled = df_unlabeled.copy()
    print("\n" + "=" * 60 + f"\nTEACHER TRANSFORMER — {MODEL_NAME}\n" + "=" * 60)

    unique_sessions = [
        tuple(x) for x in
        df_labeled[df_labeled[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
    ]

    default_wc = WINDOW_CONFIGS["default"]
    validate_teacher_cfg(
        teacher_cfg_from_params(teacher_params, epochs=default_wc["epochs"]),
        default_wc["window_size"],
    )
    for wc in WINDOW_CONFIGS.values():
        validate_teacher_cfg(
            teacher_cfg_from_params(teacher_params, epochs=wc["epochs"]),
            wc["window_size"],
        )

    S_SIZE = default_wc["step_size"]

    teacher_dir = MODELS_DIR / "Teacher"
    soft_dir = BASE_DIR / "soft_targets" / MODEL_NAME
    plots_dir = BASE_DIR / "training_curves" / "Teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    soft_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_teacher_metrics = []

    for test_idx, (test_part, test_sess) in enumerate(unique_sessions):
        print(f"\n--- TEACHER FOLD {test_idx+1}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")

        fold_wc = WINDOW_CONFIGS.get(test_part, WINDOW_CONFIGS["default"])
        W_SIZE = fold_wc["window_size"]
        cfg = teacher_cfg_from_params(teacher_params, epochs=fold_wc["epochs"])

        df_train = df_labeled[~((df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess))].copy()
        df_test  = df_labeled[ (df_labeled[COL_PARTICIPANT] == test_part)  & (df_labeled[COL_SESSION] == test_sess)].copy()

        scaler = RobustScaler()
        df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
        df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])
        df_unlabeled_train = df_unlabeled.copy()
        if len(df_unlabeled_train) > 0:
            df_unlabeled_train[SIGNAL_COLS] = scaler.transform(df_unlabeled_train[SIGNAL_COLS])

        X_train, y_train = extract_all_windows(df_train, W_SIZE, S_SIZE)
        X_test, y_test = extract_all_windows(df_test, W_SIZE, S_SIZE)
        X_unlabeled, _ = (
            extract_all_windows(df_unlabeled_train, W_SIZE, S_SIZE)
            if len(df_unlabeled_train) > 0 else (np.array([]), np.array([]))
        )

        if len(X_test) == 0:
            print(f"  WARNING: test vide. Skip.")
            continue

        print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Unlabeled: {len(X_unlabeled)} fenêtres")

        w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        weight_dict = dict(zip(np.unique(y_train), w))

        X_train_fit, y_train_fit = X_train, y_train
        teacher = None
        try:
            teacher = build_teacher(W_SIZE, len(SIGNAL_COLS), num_classes, cfg)

            history = teacher.fit(
                X_train, to_categorical(y_train, num_classes),
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                validation_split=0.1,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                    TerminateOnNaN(),
                ],
                class_weight=weight_dict,
                verbose=1,
            )

            inf_model = build_teacher_inference_model(teacher)
            X_train_v2, y_train_v2, pseudo_y, _ = add_pseudo_labels(
                inf_model, X_train, y_train, X_unlabeled
            )
            if len(pseudo_y) > 0:
                w_v2 = compute_class_weight("balanced", classes=np.unique(y_train_v2), y=y_train_v2)
                weight_dict_v2 = dict(zip(np.unique(y_train_v2), w_v2))
                _free_memory(teacher)
                teacher = build_teacher(W_SIZE, len(SIGNAL_COLS), num_classes, cfg)
                history = teacher.fit(
                    X_train_v2, to_categorical(y_train_v2, num_classes),
                    epochs=cfg["epochs"],
                    batch_size=cfg["batch_size"],
                    validation_split=0.1,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                        TerminateOnNaN(),
                    ],
                    class_weight=weight_dict_v2,
                    verbose=1,
                )
                X_train_fit, y_train_fit = X_train_v2, y_train_v2

            y_pred = _predict_classes(teacher, X_test, cfg["batch_size"])
            f1_mac = f1_score(y_test, y_pred, average="macro")
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  ✅ Teacher F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            plot_fold(history, test_idx + 1, test_part, test_sess,
                      plots_dir, f1_mac, bal_acc, y_test, y_pred)

            all_teacher_metrics.append({
                "Fold": test_idx + 1, "Participant": int(test_part), "Session": int(test_sess),
                "F1_Macro": float(f1_mac), "Balanced_Accuracy": float(bal_acc),
            })

            fold_stem = f"fold{test_idx+1:02d}_P{test_part}_S{test_sess}"
            fold_teacher_path = teacher_dir / f"{fold_stem}_teacher.keras"
            teacher.save(fold_teacher_path)
            print(f"  Teacher fold sauvegardé : {fold_teacher_path.name}")

            print(f"  Génération des soft targets (T={DISTILL_TEMP})...")
            logits_train, soft_train = generate_soft_targets(
                teacher, X_train_fit, DISTILL_TEMP, cfg["batch_size"]
            )
            logits_test, soft_test = generate_soft_targets(teacher, X_test, DISTILL_TEMP, cfg["batch_size"])
            save_soft_targets(soft_train, logits_train, y_train_fit, soft_dir / f"{fold_stem}_train.npz")
            save_soft_targets(soft_test, logits_test, y_test, soft_dir / f"{fold_stem}_test.npz")

        except Exception as exc:
            print(f"  ❌ Teacher fold échoué ({type(exc).__name__}: {exc})")
        finally:
            _free_memory(teacher)
            del X_train, X_test, y_train, y_test
            gc.collect()

    if not all_teacher_metrics:
        print("Aucun fold Teacher complété.")
        return None

    df_res = pd.DataFrame(all_teacher_metrics)
    print("\n" + "=" * 60 + f"\nRÉSULTATS TEACHER — {MODEL_NAME}\n" + "=" * 60)
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
        "params": teacher_params,
        "temperature": DISTILL_TEMP,
        "folds": all_teacher_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(curr_metrics, f, indent=4)
    print(f"\n📊 Métriques Teacher → {METRICS_PATH}")

    print("\n" + "=" * 60 + "\nTEACHER FINAL — Entraînement sur toutes les données\n" + "=" * 60)
    W_SIZE = default_wc["window_size"]
    cfg = teacher_cfg_from_params(teacher_params, epochs=default_wc["epochs"])

    df_all = df_labeled.copy()
    scaler = RobustScaler()
    df_all[SIGNAL_COLS] = scaler.fit_transform(df_all[SIGNAL_COLS])
    df_unlabeled_all = df_unlabeled.copy()
    if len(df_unlabeled_all) > 0:
        df_unlabeled_all[SIGNAL_COLS] = scaler.transform(df_unlabeled_all[SIGNAL_COLS])

    X_all, y_all = extract_all_windows(df_all, W_SIZE, S_SIZE)
    X_unlabeled_all, _ = (
        extract_all_windows(df_unlabeled_all, W_SIZE, S_SIZE)
        if len(df_unlabeled_all) > 0 else (np.array([]), np.array([]))
    )
    print(f"  Fenêtres labellisées : {len(X_all)} | unlabeled : {len(X_unlabeled_all)}")

    teacher_final = build_teacher(W_SIZE, len(SIGNAL_COLS), num_classes, cfg)
    w_all = compute_class_weight("balanced", classes=np.unique(y_all), y=y_all)
    teacher_final.fit(
        X_all, to_categorical(y_all, num_classes),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=0.05,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            TerminateOnNaN(),
        ],
        class_weight=dict(zip(np.unique(y_all), w_all)),
        verbose=1,
    )

    inf_final = build_teacher_inference_model(teacher_final)
    X_all_v2, y_all_v2, pseudo_y, _ = add_pseudo_labels(inf_final, X_all, y_all, X_unlabeled_all)
    if len(pseudo_y) > 0:
        w_all_v2 = compute_class_weight("balanced", classes=np.unique(y_all_v2), y=y_all_v2)
        _free_memory(teacher_final)
        teacher_final = build_teacher(W_SIZE, len(SIGNAL_COLS), num_classes, cfg)
        teacher_final.fit(
            X_all_v2, to_categorical(y_all_v2, num_classes),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            validation_split=0.05,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                TerminateOnNaN(),
            ],
            class_weight=dict(zip(np.unique(y_all_v2), w_all_v2)),
            verbose=1,
        )
        X_all, y_all = X_all_v2, y_all_v2

    teacher_path = teacher_dir / f"{MODEL_NAME}_final.keras"
    teacher_final.save(teacher_path)
    print(f"  ✅ Teacher final sauvegardé : {teacher_path}")

    print("  Génération des soft targets globaux...")
    logits_all, soft_all = generate_soft_targets(teacher_final, X_all, DISTILL_TEMP, cfg["batch_size"])
    global_soft_path = teacher_dir / f"{MODEL_NAME}_soft_targets_global.npz"
    save_soft_targets(soft_all, logits_all, y_all, global_soft_path)
    _free_memory(teacher_final)
    del X_all, y_all
    gc.collect()

    return global_soft_path

# ══════════════════════════════════════════════════════════════════════
# 11. UTILITAIRES DE FENÊTRAGE (identiques au CNN-D1)
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

# ══════════════════════════════════════════════════════════════════════
# 12. MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60 + f"\nTRAIN TEACHER — {MODEL_NAME}\n" + "=" * 60)
    if not DATA_MODEL_READY.exists():
        return print(f"❌ Dataset non trouvé : {DATA_MODEL_READY}")

    df = pd.read_csv(DATA_MODEL_READY)
    df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors="coerce").fillna(-1).astype(int)
    df_labeled = df[df[COL_LABEL] >= 0].copy()
    df_unlabeled = df[df[COL_LABEL] == -1].copy()
    print(f"Labeled: {len(df_labeled)} lignes | Unlabeled: {len(df_unlabeled)} lignes")

    num_classes = len(LABEL_MAPPING)

    teacher_params = (
        optimize_hyperparams(df_labeled, num_classes)
        if USE_OPTUNA_DISTILL_TEACHER
        else MODEL_PARAMS["DISTILL_TEACHER"]
    )

    train_teacher_loso(df_labeled, num_classes, teacher_params, df_unlabeled)

if __name__ == "__main__":
    main()