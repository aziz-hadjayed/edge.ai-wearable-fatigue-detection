# ------------------------------------------// Importations
import pandas as pd

from src.config import DATA_RAW

# ------------------------------------------// param

path = DATA_RAW


# ------------------------------------------// Fonctions de préparation des données
def load_raw_data(path):
    """Charge les données brutes"""
    return pd.read_csv(path)


def clean_data(df):
    """Supprime les valeurs manquantes, duplicatas"""
    return df.dropna().drop_duplicates()


def encode_features(df):
    """Encode les variables catégoriques"""
    # One-hot encoding, Label encoding, etc.
    return df_encoded


def normalize_features(df):
    """Normalise les données (0-1 ou Z-score)"""
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(df)
