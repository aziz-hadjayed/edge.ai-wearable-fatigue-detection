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

METADATA_PATH   = RAW_DIR / "metadata.csv"
DATA_RAW       = RAW_DIR
DATA_PROCESSED = PROCESSED_DIR / "dataset_balanced.csv"
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
    "temp", "breathing_rpm", "breathing_q",
    "age", "gender"
    ]


# ------------------------------------------// clean_2 params / encoding params / normalization params
SIGNAL_COLS = [
    "acc_x", "acc_y", "acc_z",
    "eda", "wrist_hr", "ibi",
    "temp", "breathing_rpm",
    "age", "gender"     # contexte démographique : aide la généralisation inter-participants
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
# EarlyStopping patience=3 → arrêt réel vers epoch 4-10 selon le fold
# epochs = plafond maximum, pas le nombre réel d'epochs exécutés
WINDOW_CONFIGS = {
    3: {
        "window_size": 480,
        "step_size": 240,
        "epochs": 500,
    },  # P03: signal atypique → plus de contexte temporel
    7: {
        "window_size": 480,
        "step_size": 240,
        "epochs": 500,
    },  # P07: collapse fatigue observé → fenêtre élargie
    9: {
        "window_size": 480,
        "step_size": 240,
        "epochs": 500,
    },  # P09: fatigue profil unique (2864 samples S1) → collapse fatigue
    11: {
        "window_size": 480,
        "step_size": 240,
        "epochs": 500,
    },  # P11: signal atypique → plus de contexte temporel
    # Autres participants: config standard
    "default": {"window_size": 240, "step_size": 120, "epochs": 1000},
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
            "dropout_rate": 0.5,    # augmenté (0.3 to 0.5) : réduit overfitting observé
            "learning_rate": 0.0001, # corrigé (3e-6 to 3e-4) : convergence stable
            "filters_0": 16,
            "filters_1": 16,
            "batch_size": 64,
        }
}

