import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from pathlib import Path
import sys

# Ajouter le root au path pour les imports src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import *

def apply_smote():
    print("🚀 Chargement du dataset pour équilibrage...")
    df = pd.read_csv(DATA_PROCESSED)
    
    print(f"📊 Distribution avant SMOTE :\n{df[COL_LABEL].value_counts()}")
    
    # Colonnes de features (signaux + age/gender) 
    #Note: on exclut timestamp, participant, session car SMOTE crée des valeurs continues
    features = [c for c in SIGNAL_COLS if c in df.columns]
    
    X = df[features]
    y = df[COL_LABEL]
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Reconstruire le DataFrame
    df_balanced = pd.DataFrame(X_res, columns=features)
    df_balanced[COL_LABEL] = y_res
    
    # Pour les colonnes exclues (participant, session, timestamp), on met des valeurs par défaut
    # car les lignes synthétiques ne correspondent à aucune session réelle.
    df_balanced[COL_PARTICIPANT] = 0
    df_balanced[COL_SESSION] = 0
    df_balanced[COL_TIMESTAMP] = 0
    
    # On tente de restaurer les vraies valeurs pour les lignes originales
    # (SMOTE garde les lignes originales en premier généralement, mais il vaut mieux reconstruire proprement)
    # En réalité, SMOTE ne garantit pas l'ordre original après merge.
    
    output_path = Path(DATA_PROCESSED).parent / "dataset_balanced.csv"
    df_balanced.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset équilibré sauvegardé dans : {output_path}")
    print(f"📊 Distribution après SMOTE :\n{df_balanced[COL_LABEL].value_counts()}")
    print(f"📈 Taille finale : {len(df_balanced)} lignes")

if __name__ == "__main__":
    apply_smote()
