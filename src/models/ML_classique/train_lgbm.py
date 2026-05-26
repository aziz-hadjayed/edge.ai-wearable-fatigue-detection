import gc
import json
import sys
import warnings
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# 1. IMPORTS
# ══════════════════════════════════════════════════════════════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import *

from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

# ══════════════════════════════════════════════════════════════════════
# 2. CONSTANTES
# ══════════════════════════════════════════════════════════════════════
MODEL_NAME        = "LGBM_WITH_64HZ"
LABEL_MAPPING     = {-1: 0, 0: 1, 1: 2}
TARGET_NAMES      = ["baseline (-1)", "activity (0)", "fatigue (1)"]
N_OPTUNA_SESSIONS = 36      # sessions tirées au hasard pour évaluer chaque trial Optuna

WINDOW_SIZE = WINDOW_CONFIGS["default"]["window_size"]
STEP_SIZE   = WINDOW_CONFIGS["default"]["step_size"]

STATS        = ["mean", "std", "min", "max", "median"]
FEATURE_NAMES = [f"{col}_{stat}" for col in SIGNAL_COLS for stat in STATS]

OPTUNA_SPACE = {
    "n_estimators":      (100, 1000),
    "learning_rate":     (1e-3, 0.3),
    "num_leaves":        (15, 127),
    "max_depth":         [-1, 6, 10, 15],
    "min_child_samples": (5, 50),
    "subsample":         (0.5, 1.0),
    "colsample_bytree":  (0.5, 1.0),
    "reg_alpha":         (1e-4, 10.0),
    "reg_lambda":        (1e-4, 10.0),
}

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
        "n_estimators":      trial.suggest_int("n_estimators",      *OPTUNA_SPACE["n_estimators"]),
        "learning_rate":     trial.suggest_float("learning_rate",   *OPTUNA_SPACE["learning_rate"], log=True),
        "num_leaves":        trial.suggest_int("num_leaves",        *OPTUNA_SPACE["num_leaves"]),
        "max_depth":         trial.suggest_categorical("max_depth",  OPTUNA_SPACE["max_depth"]),
        "min_child_samples": trial.suggest_int("min_child_samples", *OPTUNA_SPACE["min_child_samples"]),
        "subsample":         trial.suggest_float("subsample",       *OPTUNA_SPACE["subsample"]),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",*OPTUNA_SPACE["colsample_bytree"]),
        "reg_alpha":         trial.suggest_float("reg_alpha",       *OPTUNA_SPACE["reg_alpha"], log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",      *OPTUNA_SPACE["reg_lambda"], log=True),
        "n_jobs": -1, "random_state": 42, "verbose": -1,
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
# 9. MODÈLE GLOBAL
# ══════════════════════════════════════════════════════════════════════
def train_global_model(df, lgbm_params):
    import joblib
    print("\n" + "=" * 60 + f"\nMODÈLE GLOBAL — {MODEL_NAME}\n" + "=" * 60)
    X_all, y_all = extract_all_windows(df)
    print(f"  Fenêtres totales : {len(X_all)}")

    w = compute_class_weight("balanced", classes=np.unique(y_all), y=y_all)
    sample_weights = np.array([w[int(yi)] for yi in y_all])

    model = None
    try:
        model = build_model(lgbm_params)
        model.fit(X_all, y_all, sample_weight=sample_weights)
        models_dir = MODELS_DIR / "LGBM"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_dir / f"{MODEL_NAME}_global.pkl")
        print(f"  ✅ Modèle global sauvegardé : {MODEL_NAME}_global.pkl")
    except Exception as exc:
        print(f"  ❌ Modèle global échoué ({type(exc).__name__}: {exc})")
    finally:
        _free_memory(model)
        if 'X_all' in locals(): del X_all
        if 'y_all' in locals(): del y_all
        gc.collect()

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

    best_params = (
        optimize_hyperparams(df, num_classes)
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

        df_train = df[~((df[COL_PARTICIPANT] == test_part) & (df[COL_SESSION] == test_sess))].copy()
        df_test  = df[ (df[COL_PARTICIPANT] == test_part)  & (df[COL_SESSION] == test_sess)].copy()

        model = None
        try:
            X_train, y_train = extract_all_windows(df_train)
            X_test,  y_test  = extract_all_windows(df_test)

            if len(X_test) == 0:
                print(f"  WARNING: test vide pour P{test_part}. Skip."); continue

            print(f"  Train: {len(X_train)} | Test: {len(X_test)} fenêtres")

            classes = np.unique(y_train)
            w = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weight_map = dict(zip(classes, w))
            sample_weights = np.array([class_weight_map[yi] for yi in y_train])

            model = build_model(lgbm_params)
            model.fit(X_train, y_train, sample_weight=sample_weights)

            y_pred  = model.predict(X_test)
            f1_mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  ✅ F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

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
            print(f"  ❌ Fold {fold_idx+1} échoué ({type(exc).__name__}: {exc})")
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
    print(f"\n📊 Métriques → {METRICS_PATH}")

    train_global_model(df, lgbm_params)


if __name__ == "__main__":
    main()
