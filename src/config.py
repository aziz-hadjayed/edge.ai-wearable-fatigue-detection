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

# ------------------------------------------// Training CNN 1D
USE_OPTUNA_CNN_1D = False   # True → recherche Optuna | False → hyperparamètres par défaut
OPTUNA_PATH_CNN_D1    = BASE_DIR / "reports" / "optuna_cnn_results.json"
BATCH_SIZE_CNN_1D = 64

# LOSO CNN ADAPTATIF - Configuration par participant
# Participants difficiles (F1 < 60%) = fenêtre + epochs augmentés
WINDOW_CONFIGS = {
    3: {
        "window_size": 480,
        "step_size": 240,
        "epochs": 50,
    },  # P03: difficile (41% F1) → besoin + contexte
    7: {
        "window_size": 480,
        "step_size": 240,
        "epochs": 50,
    },  # P07: difficile (55% F1) → besoin + contexte
    11: {
        "window_size": 480,
        "step_size": 240,
        "epochs": 50,
    },  # P11: difficile (54% F1) → besoin + contexte
    # Autres participants: config standard
    "default": {"window_size": 240, "step_size": 120, "epochs": 30},
}




# ------------------------------------------// Hyperparamètres
MODEL_PARAMS = {
    "RandomForest": {"n_estimators": 100, "max_depth": 10},
    "SVM": {"kernel": "rbf", "C": 1.0},
    "CNN_1D": {
            "n_conv_blocks": 2,
            "kernel_size": 3,
            "use_batchnorm": False,
            "dense_units": 128,
            "dropout_rate": 0.3,
            "learning_rate": 1e-3,
            "filters_0": 32,
            "filters_1": 64,
            "batch_size": 64,
        }
}

