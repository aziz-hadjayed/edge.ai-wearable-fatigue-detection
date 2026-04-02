import sys
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from pathlib import Path
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 240   # 60s @ 4Hz
STEP_SIZE   = 120   # 50% overlap

# Labels: -1 → 0, 0 → 1, 1 → 2  (LightGBM requiert classes 0..N-1)
LABEL_MAPPING  = {-1: 0, 0: 1, 1: 2}
TARGET_NAMES   = ["baseline (-1)", "activity (0)", "fatigue (1)"]
MODEL_NAME     = "LGBM_LOSO"

LGBM_PARAMS = {
    "objective":        "multiclass",
    "num_class":        3,
    "metric":           "multi_logloss",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "n_jobs":           -1,
    "random_state":     42,
    "verbose":          -1,
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction  (stats par fenêtre glissante)
# ─────────────────────────────────────────────────────────────────────────────
STATS = ["mean", "std", "min", "max", "median"]

FEATURE_NAMES = [f"{col}_{stat}" for col in SIGNAL_COLS for stat in STATS]


def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    """window shape : (WINDOW_SIZE, n_signals) → vecteur 1D de features."""
    feats = []
    for i in range(window.shape[1]):
        s = window[:, i]
        feats += [s.mean(), s.std(), s.min(), s.max(), float(np.median(s))]
    return np.array(feats, dtype=np.float32)


def extract_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Découpe chaque groupe (participant, session) en fenêtres glissantes.
    Label = vote majoritaire sur la fenêtre.
    Retourne (X_features, y) sans chevauchement inter-sessions.
    """
    X_all, y_all = [], []

    df = df.sort_values([COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP])

    for (_, _), group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        X_raw = group[SIGNAL_COLS].values
        y_raw = group[COL_LABEL].values
        n = len(X_raw)

        for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
            end   = start + WINDOW_SIZE
            w_x   = X_raw[start:end]
            w_y   = y_raw[start:end]

            vals, counts = np.unique(w_y, return_counts=True)
            majority     = vals[np.argmax(counts)]

            X_all.append(extract_features_from_window(w_x))
            y_all.append(majority)

    return np.array(X_all, dtype=np.float32), np.array(y_all)


# ─────────────────────────────────────────────────────────────────────────────
# Main  — LOSO
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not DATA_PROCESSED.exists():
        print(f"Dataset introuvable : {DATA_PROCESSED}")
        return

    print(f"Chargement : {DATA_PROCESSED}")
    df = pd.read_csv(DATA_PROCESSED)
    df[COL_LABEL] = df[COL_LABEL].map(LABEL_MAPPING)

    participants = sorted(df[COL_PARTICIPANT].unique())
    print(f"{len(participants)} participants : {participants}")

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for fold_idx, test_part in enumerate(participants):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/{len(participants)}  —  TEST : participant {test_part}")
        print("="*60)

        df_train = df[df[COL_PARTICIPANT] != test_part]
        df_test  = df[df[COL_PARTICIPANT] == test_part]

        X_train, y_train = extract_windows(df_train)
        X_test,  y_test  = extract_windows(df_test)

        print(f"Fenêtres train : {len(X_train)} | test : {len(X_test)}")

        if len(X_test) == 0:
            print("SKIP — test vide")
            continue

        # Poids de classes (calculés sur le train uniquement)
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        sample_weights = np.array([weights[int(y)] for y in y_train])

        # Entraînement
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(100)],
            eval_set=[(X_test, y_test)],
        )

        # Évaluation
        y_pred   = model.predict(X_test)
        f1_mac   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        f1_wei   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        bal_acc  = balanced_accuracy_score(y_test, y_pred)

        print(f"\nF1-Macro          : {f1_mac:.4f}")
        print(f"F1-Weighted       : {f1_wei:.4f}")
        print(f"Balanced Accuracy : {bal_acc:.4f}")

        labels_present = np.unique(np.concatenate([y_test, y_pred]))
        print("\n" + classification_report(
            y_test, y_pred,
            labels=labels_present,
            target_names=[TARGET_NAMES[i] for i in labels_present],
            zero_division=0
        ))

        # Sauvegarde modèle
        model_stem = f"{MODEL_NAME}_fold{fold_idx+1}_testP{test_part}"
        model_path = MODELS_DIR / f"{model_stem}.pkl"
        joblib.dump(model, model_path)
        print(f"Modèle sauvegardé : {model_path}")

        all_metrics.append({
            "Fold":               fold_idx + 1,
            "Test_Participant":   test_part,
            "F1_Macro":           f1_mac,
            "F1_Weighted":        f1_wei,
            "Balanced_Accuracy":  bal_acc,
            "Train_Windows":      len(X_train),
            "Test_Windows":       len(X_test),
        })

    # ── Résumé global ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RÉSULTATS GLOBAUX")
    print("="*60)

    if not all_metrics:
        print("Aucun fold valide.")
        return

    df_results = pd.DataFrame(all_metrics)

    mean_row = {"Fold": "MEAN", "Test_Participant": "ALL",
                "F1_Macro":          df_results["F1_Macro"].mean(),
                "F1_Weighted":       df_results["F1_Weighted"].mean(),
                "Balanced_Accuracy": df_results["Balanced_Accuracy"].mean()}
    std_row  = {"Fold": "STD",  "Test_Participant": "ALL",
                "F1_Macro":          df_results["F1_Macro"].std(),
                "F1_Weighted":       df_results["F1_Weighted"].std(),
                "Balanced_Accuracy": df_results["Balanced_Accuracy"].std()}

    df_results = pd.concat(
        [df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True
    )

    csv_path  = reports_dir / "results_lgbm_loso.csv"
    json_path = reports_dir / "metrics_lgbm_loso.json"

    df_results.to_csv(csv_path, index=False)

    metrics_json = {
        "model":    MODEL_NAME,
        "strategy": "LOSO",
        "window_size": WINDOW_SIZE,
        "step_size":   STEP_SIZE,
        "features":    FEATURE_NAMES,
        "lgbm_params": LGBM_PARAMS,
        "summary": {
            "F1_Macro_mean":          float(mean_row["F1_Macro"]),
            "F1_Macro_std":           float(std_row["F1_Macro"]),
            "F1_Weighted_mean":       float(mean_row["F1_Weighted"]),
            "F1_Weighted_std":        float(std_row["F1_Weighted"]),
            "Balanced_Accuracy_mean": float(mean_row["Balanced_Accuracy"]),
            "Balanced_Accuracy_std":  float(std_row["Balanced_Accuracy"]),
        },
    }
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=4)

    print(df_results[["Fold", "Test_Participant", "F1_Macro",
                       "F1_Weighted", "Balanced_Accuracy"]].to_string(index=False))
    print(f"\nCSV  → {csv_path}")
    print(f"JSON → {json_path}")


if __name__ == "__main__":
    main()
