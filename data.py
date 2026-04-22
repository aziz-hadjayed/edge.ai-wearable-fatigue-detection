# ------------------------------------------// Importations
import json
import sys
import pandas as pd
from pathlib import Path

# Ajouter src/ au path pour trouver config.py et data_prep.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import *
from data_prep import *

# ------------------------------------------// body
if __name__ == "__main__":
    print("=" * 60)
    print("CHARGEMENT DU DATASET")
    print("=" * 60)
    print(f"Répertoire source : {DATA_RAW}")

    # ── Pipeline en 3 étapes ──────────────────────────────────────────────
    df = data_clean_1(DATA_RAW)
    
    if df is None or df.empty:
        print("Aucune donnée chargée.")
        exit(1)

    df = merge_demographics(df,METADATA_PATH)
    visualize_data(df)
    df = clean_data_2(df)
    X, y = encode_features(df)
    # X, scaler = normalize_features(X) # Pas de normalisation ici — elle est faite par fold dans les scripts LOSO pour éviter le data leakage (scaler fité sur train uniquement)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    OUTPUT_PATH_NO_SMOTE.parent.mkdir(parents=True, exist_ok=True)

    df_meta = df[[COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP]].reset_index(drop=True)
    df_final = pd.concat([df_meta, X, y.rename(COL_LABEL)], axis=1)
    df_final.to_csv(OUTPUT_PATH_NO_SMOTE, index=False)
    # ── Résumé ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"✔ Fichier sauvegardé : {OUTPUT_PATH_NO_SMOTE}")
    print(f"✔ Shape finale       : {df_final.shape}")
    print(f"✔ NaN restants       : {df_final.isna().sum().sum()}")
    print(f"✔ RAM X (float32)    : {X.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"✔ Distribution label :")
    print(y.value_counts().sort_index().to_string())


