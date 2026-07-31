import gc
import json
import sys
import traceback
import warnings
from pathlib import Path

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
# pyrefly: ignore [missing-import]
import lightgbm as lgb
# pyrefly: ignore [missing-import]
import onnxruntime as ort
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import *

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from models.semi_supervised import add_pseudo_labels
from utils.apply_smote import resample_dataframe
# pyrefly: ignore [missing-import]
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

# ══════════════════════════════════════════════════════════════════════
# 2. CONSTANTES
# ══════════════════════════════════════════════════════════════════════
MODEL_NAME        = "LGBM"
LABEL_MAPPING     = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES      = ["baseline", "activity", "pre_fatigue", "fatigue"]
N_OPTUNA_SESSIONS = 36      # sessions tirées au hasard pour évaluer chaque trial Optuna

WINDOW_SIZE = WINDOW_CONFIGS["default"]["window_size"]
STEP_SIZE   = WINDOW_CONFIGS["default"]["step_size"]

STATS        = ["mean", "std", "min", "max", "median"]
FEATURE_NAMES = [f"{col}_{stat}" for col in SIGNAL_COLS for stat in STATS]


# ══════════════════════════════════════════════════════════════════════
# 3. MÉMOIRE
# ══════════════════════════════════════════════════════════════════════
def _free_memory(model=None):
    if model is not None:
        del model
    gc.collect()

# ══════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# (Pas de courbes loss/accuracy — LGBM n'est pas epoch-based)
# ══════════════════════════════════════════════════════════════════════
def plot_fold(model, fold_idx, test_part, test_sess, save_dir,
              f1_mac, bal_acc, y_true, y_pred):
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"LGBM  |  Fold {fold_idx} — P{test_part} S{test_sess}"
        f"  |  F1-Macro: {f1_mac:.3f}  |  Bal.Acc: {bal_acc:.3f}",
        fontsize=13, fontweight="bold"
    )

    # Panneau 1 — Feature Importance (gain) top-20
    importances = model.feature_importances_
    top_n       = min(20, len(FEATURE_NAMES))
    idx_sorted  = np.argsort(importances)[-top_n:]
    axes[0].barh([FEATURE_NAMES[i] for i in idx_sorted], importances[idx_sorted], color="#4CAF50")
    axes[0].set_title("Feature Importance (gain) — Top 20")
    axes[0].set_xlabel("Importance")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Panneau 2 — Confusion Matrix
    cm     = confusion_matrix(y_true, y_pred)
    labels = [TARGET_NAMES[i] for i in np.unique(np.concatenate([y_true, y_pred]))]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels,
                ax=axes[1], linewidths=0.5, linecolor="gray")
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Prédit"); axes[1].set_ylabel("Réel")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plot_path = save_dir / f"fold{fold_idx:02d}_P{test_part}_S{test_sess}_curves.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Courbes sauvegardées : {plot_path}")

# ══════════════════════════════════════════════════════════════════════
# 6. DONNÉES — extraction de features par fenêtre glissante
# ══════════════════════════════════════════════════════════════════════
def _extract_features(window):
    feats = []
    for i in range(window.shape[1]):
        s = window[:, i]
        feats += [s.mean(), s.std(), s.min(), s.max(), float(np.median(s))]
    return np.array(feats, dtype=np.float32)

def extract_all_windows(df):
    X_all, y_all = [], []
    if COL_TIMESTAMP in df.columns:
        df = df.sort_values([COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP])
    for _, group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        X_raw = group[SIGNAL_COLS].values
        y_raw = group[COL_LABEL].values
        for start in range(0, len(X_raw) - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            window = X_raw[start:end]
            vals, counts = np.unique(y_raw[start:end], return_counts=True)
            X_all.append(_extract_features(window))
            y_all.append(vals[np.argmax(counts)])
    return np.array(X_all), np.array(y_all)

# ══════════════════════════════════════════════════════════════════════
# 7. MODÈLE + OPTUNA
# ══════════════════════════════════════════════════════════════════════
def build_model(params):
    return lgb.LGBMClassifier(**params)

def optuna_objective(trial, df, num_classes):
    params = {
        "objective":         "multiclass",
        "num_class":         num_classes,
        "metric":            "multi_logloss",
        "n_estimators":      trial.suggest_int("n_estimators",      *LGBM_OPTUNA_SPACE["n_estimators"]),
        "learning_rate":     trial.suggest_float("learning_rate",   *LGBM_OPTUNA_SPACE["learning_rate"], log=True),
        "num_leaves":        trial.suggest_int("num_leaves",        *LGBM_OPTUNA_SPACE["num_leaves"]),
        "max_depth":         trial.suggest_categorical("max_depth",  LGBM_OPTUNA_SPACE["max_depth"]),
        "min_child_samples": trial.suggest_int("min_child_samples", *LGBM_OPTUNA_SPACE["min_child_samples"]),
        "subsample":         trial.suggest_float("subsample",       *LGBM_OPTUNA_SPACE["subsample"]),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",*LGBM_OPTUNA_SPACE["colsample_bytree"]),
        "reg_alpha":         trial.suggest_float("reg_alpha",       *LGBM_OPTUNA_SPACE["reg_alpha"], log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",      *LGBM_OPTUNA_SPACE["reg_lambda"], log=True),
        #    "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    import random
    unique_sessions = [tuple(x) for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
    val_sessions    = random.sample(unique_sessions, min(N_OPTUNA_SESSIONS, len(unique_sessions)))
    scores = []

    for (val_part, val_sess) in val_sessions:
        model = None
        try:
            df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
            df_val   = df[ (df[COL_PARTICIPANT] == val_part)  & (df[COL_SESSION] == val_sess)].copy()

            # SMOTE seulement sur df_train, avant l'extraction de features par fenetre
            df_train = resample_dataframe(df_train, SIGNAL_COLS)

            X_train, y_train = extract_all_windows(df_train)
            X_val,   y_val   = extract_all_windows(df_val)
            if len(X_val) == 0: continue

            w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
            sample_weights = np.array([w[int(yi)] for yi in y_train])

            model = build_model(params)
            model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                callbacks=[lgb.early_stopping(20, verbose=False)],
                eval_set=[(X_val, y_val)],
            )
            y_pred = model.predict(X_val)
            scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        finally:
            _free_memory(model)
            if 'X_train' in locals(): del X_train
            if 'X_val'   in locals(): del X_val
            if 'y_train' in locals(): del y_train
            if 'y_val'   in locals(): del y_val
            gc.collect()

    return float(np.mean(scores)) if scores else 0.0

def optimize_hyperparams(df, num_classes, n_trials=20):
    print(f"\nOPTUNA LGBM — {n_trials} trials | {N_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, df, num_classes),
        n_trials=n_trials, show_progress_bar=True,
        gc_after_trial=True, catch=(Exception,),
    )
    print(f"\nBest F1-Macro : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    OPTUNA_PATH_LGBM.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_LGBM, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return study.best_params


# ══════════════════════════════════════════════════════════════════════
# 8. UTILS — EXPORT ONNX → TFLite INT8 + HEADER C
# ══════════════════════════════════════════════════════════════════════
def _export_onnx(model, X_sample, save_path):
    """Exporte un modèle LGBM entraîné vers ONNX via onnxmltools."""
    # pyrefly: ignore [missing-import]
    import onnxmltools
    # pyrefly: ignore [missing-import]
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=15,
    )
    with open(save_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  ONNX exporté : {save_path}")

def _export_tflite_float32(onnx_path, X_repr, save_dir, model_name):
    """Récupère le TFLite float32 généré par onnx2tf (STM32H7 DPFPU supporte float32 natif)."""
    import shutil
    # pyrefly: ignore [missing-import]
    from onnx2tf import convert

    saved_model_dir = save_dir / f"{model_name}_saved_model"

    # ONNX → TFLite (onnx2tf génère float32 et float16)
    convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        output_signaturedefs=True,
        not_use_onnxsim=True,
    )

    # Float32 = format natif STM32H7 Cortex-M7 DPFPU
    tflite_float32 = saved_model_dir / f"{model_name}_float32.tflite"
    
    if not tflite_float32.exists():
        raise FileNotFoundError(f"TGLite float32 non généré: {tflite_float32}")

    tflite_path = save_dir / f"{model_name}.tflite"
    shutil.copy(tflite_float32, tflite_path)
    print(f"  TFLite float32 exporté (STM32H7 DPFPU natif) : {tflite_path}")

    # Cleanup
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    return tflite_path


def _export_c_array(tflite_path, save_dir, model_name):
    """Génère un header C array à partir du .tflite."""
    tflite_path = Path(tflite_path)
    with open(tflite_path, "rb") as f:
        tflite_bytes = f.read()
    
    hex_array = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
    
    header_content = f"""/* Auto-generated TFLite model for {model_name} */
#ifndef {model_name.upper()}_MODEL_H
#define {model_name.upper()}_MODEL_H

#include <stdint.h>

const unsigned int {model_name.lower()}_model_len = {len(tflite_bytes)};
const uint8_t {model_name.lower()}_model_data[{len(tflite_bytes)}] = {{
    {hex_array}
}};

#endif /* {model_name.upper()}_MODEL_H */
"""
    header_path = save_dir / f"{model_name}.h"
    with open(header_path, "w") as f:
        f.write(header_content)
    print(f"  Header C exporté : {header_path}")


def _export_c_array_from_bytes(file_path, save_dir, model_name):
    """Génère un header C array à partir d'un fichier binaire."""
    file_path = Path(file_path)
    with open(file_path, "rb") as f:
        model_bytes = f.read()

    hex_array = ", ".join(f"0x{b:02x}" for b in model_bytes)

    header_content = f"""/* Auto-generated TFLite model for {model_name} */
#ifndef {model_name.upper()}_MODEL_H
#define {model_name.upper()}_MODEL_H

#include <stdint.h>

const unsigned int {model_name.lower()}_model_len = {len(model_bytes)};
const uint8_t {model_name.lower()}_model_data[{len(model_bytes)}] = {{
    {hex_array}
}};

#endif /* {model_name.upper()}_MODEL_H */
"""
    header_path = save_dir / f"{model_name}.h"
    with open(header_path, "w") as f:
        f.write(header_content)
    print(f"  Header C exporté : {header_path}")


def _compute_classification_metrics(y_true, y_pred):
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_fatigue": float(precision_score(y_true, y_pred, labels=[3], average="macro", zero_division=0)),
        "recall_fatigue": float(recall_score(y_true, y_pred, labels=[3], average="macro", zero_division=0)),
    }


def _predict_tflite_float32(tflite_path, X_data):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    predictions = []
    for sample in X_data:
        x_input = np.asarray(sample, dtype=np.float32)[np.newaxis, :]
        interpreter.set_tensor(input_index, x_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        predictions.append(int(np.argmax(output, axis=-1)[0]))

    return np.array(predictions)


def _predict_onnx(onnx_path, X_data):
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X_data.astype(np.float32)})

    first_output = np.asarray(outputs[0])
    if first_output.ndim == 1:
        y_pred = first_output
    elif first_output.ndim == 2:
        y_pred = np.argmax(first_output, axis=1)
    else:
        probability_tensor = np.asarray(outputs[1])
        y_pred = np.argmax(probability_tensor, axis=1)

    return np.asarray(y_pred, dtype=int)


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
    print(f"  Comparaison edge sauvegardée : {METRICS_PATH}")


# ══════════════════════════════════════════════════════════════════════
# 9. MODÈLE GLOBAL
# ══════════════════════════════════════════════════════════════════════
def train_global_model(df_labeled, df_unlabeled, lgbm_params, num_classes):
    print("\n" + "=" * 60 + f"\nMODÈLE GLOBAL — {MODEL_NAME}\n" + "=" * 60)
    print(
        f"  Export ONNX → TFLite INT8. "
        f"Entraînement sur {len(df_labeled)} labellisées, "
        f"{len(df_unlabeled)} non-labellisées."
    )

    df_unl = df_unlabeled.copy()
    X_unlabeled, _ = (
        extract_all_windows(df_unl)
        if len(df_unl) > 0 else (np.array([]), np.array([]))
    )

    # Split 90/10 par session (pas par ligne), pour eviter toute fuite participant/session
    unique_sessions_all = [
        tuple(x) for x in df_labeled[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
    ]
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(unique_sessions_all))
    split_point = int(0.9 * len(unique_sessions_all))
    train_sessions = [unique_sessions_all[i] for i in shuffled[:split_point]]
    val_sessions   = [unique_sessions_all[i] for i in shuffled[split_point:]]

    df_tr_raw = df_labeled[
        df_labeled.set_index([COL_PARTICIPANT, COL_SESSION]).index.isin(train_sessions)
    ].copy()
    df_vl_raw = df_labeled[
        df_labeled.set_index([COL_PARTICIPANT, COL_SESSION]).index.isin(val_sessions)
    ].copy()

    # SMOTE seulement sur le train, avant extraction de features
    df_tr_raw = resample_dataframe(df_tr_raw, SIGNAL_COLS)

    X_tr, y_tr = extract_all_windows(df_tr_raw)
    X_vl, y_vl = extract_all_windows(df_vl_raw)
    X_all = np.concatenate([X_tr, X_vl], axis=0)
    y_all = np.concatenate([y_tr, y_vl], axis=0)
    del df_tr_raw, df_vl_raw, df_unl
    gc.collect()
    print(f"  Fenêtres labellisées : {len(X_all)} | unlabeled : {len(X_unlabeled)}")

    model = None
    X_repr = X_all
    try:
        # Poids de classe
        w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        weight_dict = dict(zip(np.unique(y_tr), w))
        sample_weights = np.array([weight_dict[yi] for yi in y_tr])

        # Entraînement initial
        model = build_model(lgbm_params)
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_weights,
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )

        # Pseudo-labeling
        X_tr_v2, y_tr_v2, pseudo_y, _ = add_pseudo_labels(model, X_tr, y_tr, X_unlabeled)
        if len(pseudo_y) > 0:
            print(f"  Pseudo-labels générés : {len(pseudo_y)} fenêtres")
            w_v2 = compute_class_weight("balanced", classes=np.unique(y_tr_v2), y=y_tr_v2)
            weight_dict_v2 = dict(zip(np.unique(y_tr_v2), w_v2))
            sample_weights_v2 = np.array([weight_dict_v2[yi] for yi in y_tr_v2])

            _free_memory(model)
            model = build_model(lgbm_params)
            model.fit(
                X_tr_v2, y_tr_v2,
                sample_weight=sample_weights_v2,
                eval_set=[(X_vl, y_vl)],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            X_repr = np.concatenate([X_tr_v2, X_vl], axis=0)
        else:
            print("  Aucun pseudo-label confident, modèle initial conservé.")
            X_repr = X_all

        # EXPORT : ONNX → Header C  ---------------------------------------------------->>>
        models_dir = MODELS_DIR / "LGBM"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Sauvegarde pickle backup
        pkl_path = models_dir / f"{MODEL_NAME}_global.pkl"
        joblib.dump(model, pkl_path)
        print(f"  Pickle backup : {pkl_path}")

        # 2. Export ONNX
        onnx_path = models_dir / f"{MODEL_NAME}_global.onnx"
        _export_onnx(model, X_repr[:1], onnx_path)

        # 3. ONNX → Header C array
        _export_c_array_from_bytes(onnx_path, models_dir, f"{MODEL_NAME}_global")

        # 4. Comparaison native LGBM vs ONNX
        edge_metrics = {}
        y_pred_native = model.predict(X_vl)
        edge_metrics["native_lgbm"] = _compute_classification_metrics(y_vl, y_pred_native)
        edge_metrics["native_lgbm"]["model_size_kb"] = round(pkl_path.stat().st_size / 1024, 2)

        try:
            y_pred_onnx = _predict_onnx(onnx_path, X_vl)
            edge_metrics["onnx"] = _compute_classification_metrics(y_vl, y_pred_onnx)
            edge_metrics["onnx"]["model_size_kb"] = round(onnx_path.stat().st_size / 1024, 2)
        except Exception as exc:
            print(f"  Comparaison ONNX échouée ({type(exc).__name__}: {exc})")
            traceback.print_exc()

        _save_edge_comparison_metrics(MODEL_NAME, edge_metrics)

        # 5. Feature importance globale
        importances = model.feature_importances_
        top_n = min(30, len(FEATURE_NAMES))
        idx_sorted = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            [FEATURE_NAMES[i] for i in idx_sorted],
            importances[idx_sorted],
            color="#2196F3"
        )
        ax.set_title(f"Feature Importance Globale — {MODEL_NAME} (Top {top_n})")
        ax.set_xlabel("Importance (gain)")
        ax.grid(True, alpha=0.3, axis="x")

        fi_path = BASE_DIR / "training_curves" / "LGBM" / "global_feature_importance.png"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fi_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Feature importance : {fi_path}")

    except Exception as exc:
        print(f"  Modèle global échoué ({type(exc).__name__}: {exc})")
        traceback.print_exc()
        raise
    finally:
        _free_memory(model)
        del X_all, y_all
        gc.collect()

# ══════════════════════════════════════════════════════════════════════
# 10. BOUCLE LOSO
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
        if USE_OPTUNA_LGBM
        else MODEL_PARAMS["LGBM"]
    )

    # Préparer les params LightGBM finaux (ajouter les clés fixes)
    lgbm_params = {
        "objective": "multiclass", "num_class": num_classes,
        "metric": "multi_logloss", "n_jobs": -1,
        "random_state": 42, "verbose": -1,
        **best_params,
    }

    all_metrics = []
    import random
    random.seed(42)
    plots_dir   = BASE_DIR / "training_curves" / "LGBM"
    models_dir  = MODELS_DIR / "LGBM"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (test_part, test_sess) in enumerate(unique_sessions):
        print(f"\n--- FOLD {fold_idx+1}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")

        df_train = df_labeled[~((df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess))].copy()
        df_test  = df_labeled[ (df_labeled[COL_PARTICIPANT] == test_part)  & (df_labeled[COL_SESSION] == test_sess)].copy()

        model = None
        try:
            # SMOTE seulement sur df_train, avant l'extraction de features par fenetre
            df_train = resample_dataframe(df_train, SIGNAL_COLS)

            X_train, y_train = extract_all_windows(df_train)
            X_test,  y_test  = extract_all_windows(df_test)
            X_unlabeled, _ = extract_all_windows(df_unlabeled)

            if len(X_test) == 0:
                print(f"  WARNING: test vide pour P{test_part}. Skip."); continue

            print(f"  Train: {len(X_train)} | Test: {len(X_test)} fenêtres")

            classes = np.unique(y_train)
            w = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weight_map = dict(zip(classes, w))
            sample_weights = np.array([class_weight_map[yi] for yi in y_train])

            model = build_model(lgbm_params)
            model.fit(X_train, y_train, sample_weight=sample_weights)

            X_train_v2, y_train_v2, pseudo_y, _ = add_pseudo_labels(model, X_train, y_train, X_unlabeled)
            if len(pseudo_y) > 0:
                classes_v2 = np.unique(y_train_v2)
                w_v2 = compute_class_weight("balanced", classes=classes_v2, y=y_train_v2)
                class_weight_map_v2 = dict(zip(classes_v2, w_v2))
                sample_weights_v2 = np.array([class_weight_map_v2[yi] for yi in y_train_v2])
                model = build_model(lgbm_params)
                model.fit(X_train_v2, y_train_v2, sample_weight=sample_weights_v2)

            y_pred  = model.predict(X_test)
            f1_mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            labels_present  = np.unique(np.concatenate([y_test, y_pred]))
            cm_target_names = [TARGET_NAMES[i] for i in labels_present]
            print("\n" + classification_report(y_test, y_pred,
                  labels=labels_present, target_names=cm_target_names, zero_division=0))

            plot_fold(model, fold_idx+1, test_part, test_sess,
                      plots_dir, f1_mac, bal_acc, y_test, y_pred)

            all_metrics.append({
                "Fold": fold_idx+1, "Participant": int(test_part), "Session": int(test_sess),
                "F1_Macro": float(f1_mac), "Balanced_Accuracy": float(bal_acc),
            })

        except Exception as exc:
            print(f"  Fold {fold_idx+1} échoué ({type(exc).__name__}: {exc})")
        finally:
            _free_memory(model)
            if 'X_train' in locals(): del X_train
            if 'X_test'  in locals(): del X_test
            if 'y_train' in locals(): del y_train
            if 'y_test'  in locals(): del y_test
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
    print(f"\n Métriques → {METRICS_PATH}")

    train_global_model(df_labeled, df_unlabeled, lgbm_params, num_classes)


if __name__ == "__main__":
    main()
