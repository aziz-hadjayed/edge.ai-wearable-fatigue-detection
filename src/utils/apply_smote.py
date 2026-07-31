# src/utils/apply_smote.py
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import sys

# Ajouter le root et src au path pour les imports
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from config import *


def _timestamp_delta(df_session):
    timestamps = pd.to_numeric(df_session[COL_TIMESTAMP], errors="coerce")
    timestamps = timestamps.dropna().sort_values()
    positive_deltas = timestamps.diff().dropna()
    positive_deltas = positive_deltas[positive_deltas > 0]

    if positive_deltas.empty:
        return 0.0

    return float(positive_deltas.median())


def _assign_synthetic_timestamps(df_resampled, df_session, features, original_len):
    synthetic_count = len(df_resampled) - original_len
    if synthetic_count <= 0:
        return

    nearest_neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nearest_neighbors.fit(df_session[features])

    synthetic_features = df_resampled.loc[original_len:, features]
    _, nearest_indices = nearest_neighbors.kneighbors(synthetic_features)
    nearest_indices = nearest_indices.ravel()

    nearest_timestamps = pd.to_numeric(
        df_session.iloc[nearest_indices][COL_TIMESTAMP],
        errors="coerce",
    ).to_numpy()
    delta_t = _timestamp_delta(df_session)

    df_resampled.loc[original_len:, COL_TIMESTAMP] = nearest_timestamps + delta_t


def _apply_smote_for_session(group_key, df_session, features):
    participant_id, session_id = group_key
    y = df_session[COL_LABEL]
    label_counts = y.value_counts()
    group_name = f"participant {participant_id}, session {session_id}"

    if len(label_counts) < 2:
        print(
            f"  {group_name}: SMOTE ignore "
            f"(une seule classe: {label_counts.to_dict()})"
        )
        return df_session.copy()

    min_class_count = int(label_counts.min())
    if min_class_count < 2:
        print(
            f"  {group_name}: SMOTE ignore "
            f"(classe trop petite: {label_counts.to_dict()})"
        )
        return df_session.copy()

    k_neighbors = min(5, min_class_count - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(df_session[features], y)

    df_resampled = pd.DataFrame(X_res, columns=features)
    df_resampled[COL_LABEL] = y_res

    original_len = len(df_session)
    df_resampled[COL_PARTICIPANT] = participant_id
    df_resampled[COL_SESSION] = session_id
    df_resampled[COL_TIMESTAMP] = np.nan

    df_resampled.loc[:original_len - 1, COL_TIMESTAMP] = df_session[COL_TIMESTAMP].values
    _assign_synthetic_timestamps(df_resampled, df_session, features, original_len)

    added = len(df_resampled) - original_len
    print(
        f"  {group_name}: "
        f"{label_counts.to_dict()} -> {df_resampled[COL_LABEL].value_counts().to_dict()} "
        f"(+{added})"
    )
    return df_resampled

def resample_dataframe(df, features):
    """
    Applique SMOTE par session sur un DataFrame deja en memoire.
    Reutilisable a la fois par apply_smote() (pipeline global)
    et par les scripts de train (SMOTE local a df_fit dans une boucle LOSO).
    """
    balanced_parts = []
    group_cols = [COL_PARTICIPANT, COL_SESSION]
    for group_key, df_session in df.groupby(group_cols, sort=False):
        balanced_parts.append(
            _apply_smote_for_session(group_key, df_session, features)
        )
    df_balanced = pd.concat(balanced_parts, ignore_index=True)
    return df_balanced

def resample_windows_smote(X, y, k_neighbors=5, random_state=42):
    """
    Applique SMOTE sur des fenetres 3D (n_samples, window_size, n_features).
    Aplati chaque fenetre en vecteur 1D pour SMOTE, puis reforme les fenetres.
    Utilisee quand les donnees sont deja pre-fenetrees (session_windows),
    contrairement a resample_dataframe qui opere sur des lignes brutes.
    """
    n_samples, window_size, n_features = X.shape

    label_counts = pd.Series(y).value_counts()
    if len(label_counts) < 2:
        print(f"  SMOTE ignore (une seule classe: {label_counts.to_dict()})")
        return X, y

    min_class_count = int(label_counts.min())
    if min_class_count < 2:
        print(f"  SMOTE ignore (classe trop petite: {label_counts.to_dict()})")
        return X, y

    k = min(k_neighbors, min_class_count - 1)
    X_flat = X.reshape(n_samples, window_size * n_features)

    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res_flat, y_res = smote.fit_resample(X_flat, y)

    X_res = X_res_flat.reshape(-1, window_size, n_features)
    added = len(X_res) - n_samples
    print(f"  SMOTE : {label_counts.to_dict()} -> "
          f"{pd.Series(y_res).value_counts().to_dict()} (+{added})")

    return X_res, y_res

def apply_smote():
    print("Chargement du dataset pour equilibrage par session (sans SMOTE)...")
    input_path = OUTPUT_PATH_NO_SMOTE
    if not input_path.exists():
        input_path = DATA_PROCESSED

    print(f"Lecture de : {input_path}")
    df = pd.read_csv(input_path)
    print(f"Distribution globale avant SMOTE :\n{df[COL_LABEL].value_counts()}")

    features = [c for c in SIGNAL_COLS if c in df.columns]
    df_balanced = resample_dataframe(df, features)  # reutilise la fonction commune

    col_order = [COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP] + features + [COL_LABEL]
    df_balanced = df_balanced[col_order]

    output_path = OUTPUT_PATH_SMOTE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(output_path, index=False)

    print(f"\nDataset equilibre sauvegarde dans : {output_path}")
    print(f"Distribution globale apres SMOTE :\n{df_balanced[COL_LABEL].value_counts()}")
    print(f"Taille finale : {len(df_balanced)} lignes")

if __name__ == "__main__":
    apply_smote()
