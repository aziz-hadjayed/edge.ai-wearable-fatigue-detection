"""
Utilitaires semi-supervisés partagés.

Les lignes avec label == -1 sont considérées comme non étiquetées (unlabeled).
"""
import numpy as np
import pandas as pd

NUM_CLASSES = 4
PSEUDO_LABEL_THRESHOLD = 0.7
LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3}
TARGET_NAMES = ["baseline", "activity", "pre_fatigue", "fatigue"]


def prepare_labels(df, label_col):
    df = df.copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(-1).astype(int)
    return df


def split_labeled_unlabeled(df, label_col):
    df = prepare_labels(df, label_col)
    return df[df[label_col] >= 0].copy(), df[df[label_col] == -1].copy()


def _model_predict_proba(model, X, verbose=0):
    """Probabilités par fenêtre : Keras (softmax) ou sklearn (predict_proba)."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    probs = model.predict(X, verbose=verbose)
    probs = np.asarray(probs)
    if probs.ndim != 2:
        raise ValueError(
            "Le modèle doit renvoyer des probabilités (n_samples, n_classes). "
            "Utilisez predict_proba (sklearn) ou une sortie softmax (Keras)."
        )
    return probs


def add_pseudo_labels(model, X_labeled, y_labeled, X_unlabeled, threshold=PSEUDO_LABEL_THRESHOLD, verbose=True):
    if X_unlabeled is None or len(X_unlabeled) == 0:
        if verbose:
            print("  Pseudo-labels : aucune fenêtre unlabeled")
        return X_labeled, y_labeled, np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    probs = _model_predict_proba(model, X_unlabeled, verbose=0)
    confidence = probs.max(axis=1)
    pseudo_y = probs.argmax(axis=1).astype(np.int32)
    keep = confidence > threshold

    if verbose:
        kept = int(keep.sum())
        dist = {
            int(k): int(v)
            for k, v in zip(*np.unique(pseudo_y[keep], return_counts=True))
        } if kept else {}
        print(f"  Pseudo-labels retenus : {kept}/{len(X_unlabeled)} (seuil>{threshold}) {dist}")

    if not keep.any():
        return X_labeled, y_labeled, np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    return (
        np.concatenate([X_labeled, X_unlabeled[keep]], axis=0),
        np.concatenate([y_labeled, pseudo_y[keep]], axis=0),
        pseudo_y[keep],
        confidence[keep],
    )


def extract_windows(group_df, window_size, step_size, signal_cols, col_label):
    X = group_df[signal_cols].values
    y = group_df[col_label].values
    wx, wy = [], []
    for start in range(0, len(X) - window_size + 1, step_size):
        end = start + window_size
        vals, counts = np.unique(y[start:end], return_counts=True)
        wx.append(X[start:end])
        wy.append(vals[np.argmax(counts)])
    return wx, wy


def extract_all_windows(df, window_size, step_size):
    """Fenêtres glissantes (sessions groupées) — forme (n, window_size, n_features)."""
    from config import (
        COL_LABEL,
        COL_PARTICIPANT,
        COL_SESSION,
        COL_TIMESTAMP,
        SIGNAL_COLS,
    )

    X_all, y_all = [], []
    if COL_TIMESTAMP in df.columns:
        df = df.sort_values([COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP])
    for _, group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        wx, wy = extract_windows(group, window_size, step_size, SIGNAL_COLS, COL_LABEL)
        X_all.extend(wx)
        y_all.extend(wy)
    if not X_all:
        return np.array([]), np.array([], dtype=np.int32)
    return np.array(X_all), np.array(y_all, dtype=np.int32)


