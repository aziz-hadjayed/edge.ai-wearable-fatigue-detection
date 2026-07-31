# ------------------------------------------// Importations
import json
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import *
from data_prep import *

USE_FREQ_SYNC = False
# ------------------------------------------// body
if __name__ == "__main__":
    print("=" * 60)
    print("CHARGEMENT DU DATASET")
    print("=" * 60)
    print(f"Répertoire source : {DATA_RAW}")
    print(f"Mode : {'AVEC synchronisation frequence (4Hz)' if USE_FREQ_SYNC else 'SANS synchronisation frequence (no_sync)'}")

    # ── Étape 1 : chargement brut (avec ou sans sync de frequence) ────────
    if USE_FREQ_SYNC:
        df = data_clean_1(DATA_RAW)
        output_path = OUTPUT_PATH_NO_SMOTE
        num_cols_for_viz = NUM_COLS
    else:
        df = data_clean_1_no_sync(DATA_RAW)
        output_path = OUTPUT_PATH_NO_SMOTE_FREQ_NO_SYNC
        num_cols_for_viz = SIGNAL_COLS  # pas de breathing_q en mode no_sync

    if df is None or df.empty:
        print("Aucune donnée chargée.")
        exit(1)

    df = merge_demographics(df, METADATA_PATH)
    visualize_data(df, num_cols=num_cols_for_viz)
    df = clean_data_2(df) if USE_FREQ_SYNC else clean_data_2_no_sync(df)
    X, y = encode_features(df)
    # X, scaler = normalize_features(X) # Pas de normalisation ici — faite par fold dans les scripts LOSO

    # ── Sauvegarde ────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_meta = df.loc[X.index, [COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP]].reset_index(drop=True)
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    df_final = pd.concat([df_meta, X_reset, y_reset.rename(COL_LABEL)], axis=1)
    df_final.to_csv(output_path, index=False)

    # ── Résumé ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"✔ Fichier sauvegardé : {output_path}")
    print(f"✔ Shape finale       : {df_final.shape}")
    print(f"✔ NaN restants       : {df_final.isna().sum().sum()}")
    print(f"✔ RAM X (float32)    : {X.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"✔ Distribution label :")
    print(y.value_counts().sort_index().to_string())