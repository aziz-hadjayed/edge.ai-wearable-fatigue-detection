# ------------------------------------------// Importations
import os
import pandas as pd
import numpy as np

# ------------------------------------------// Chemins
from pathlib import Path

# ------------------------------------------// Chemins
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "01_raw"
INTERIM_DIR    = DATA_DIR / "02_interim"
PROCESSED_DIR  = DATA_DIR / "03_processed"

DATA_RAW       = RAW_DIR
DATA_PROCESSED = PROCESSED_DIR / "dataset_final.csv"
OUTPUT_PATH    = DATA_PROCESSED
MODELS_DIR     = BASE_DIR / "models_saved"

# Paramètres de split
TEST_SIZE      = 0.2
RANDOM_STATE   = 42




COL_PARTICIPANT = "participant"
COL_SESSION     = "session"
COL_TIMESTAMP   = "timestamp"
COL_LABEL       = "label"

# ------------------------------------------// visualition params
NUM_COLS = [
    "acc_x", "acc_y", "acc_z",
    "eda", "wrist_hr", "ibi",
    "temp", "breathing_rpm", "breathing_q"
    ]


# ------------------------------------------// clean_2 params / encoding params / normalization params
SIGNAL_COLS = [
    "acc_x", "acc_y", "acc_z",
    "eda", "wrist_hr", "ibi",
    "temp", "breathing_rpm"
]

LABEL_MAP = {
    "baseline": -1,
    "activity":  0,
    "fatigue":   1
}

# ------------------------------------------// Hyperparamètres
MODEL_PARAMS = {
    "RandomForest": {"n_estimators": 100, "max_depth": 10},
    "SVM": {"kernel": "rbf", "C": 1.0},
}

