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


def add_pseudo_labels(model, X_labeled, y_labeled, X_unlabeled, threshold=PSEUDO_LABEL_THRESHOLD, verbose=True):
    if X_unlabeled is None or len(X_unlabeled) == 0:
        if verbose:
            print("  Pseudo-labels : aucune fenêtre unlabeled")
        return X_labeled, y_labeled, np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    probs = model.predict(X_unlabeled, verbose=0)
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
