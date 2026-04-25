from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# 1. CHEMINS
# ══════════════════════════════════════════════════════════════════════
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "01_raw"
INTERIM_DIR   = DATA_DIR / "02_interim"
PROCESSED_DIR = DATA_DIR / "03_processed"
REPORTS_DIR   = BASE_DIR / "reports"
MODELS_DIR    = BASE_DIR / "models_saved"
OPTUNA = BASE_DIR / "optuna"
DATA_RAW             = RAW_DIR
METADATA_PATH        = RAW_DIR / "metadata.csv"
DATA_MODEL_READY     = DATA_DIR / "used" / "dataset.csv"
OUTPUT_PATH_NO_SMOTE = PROCESSED_DIR / "dataset_no_smote.csv"
OUTPUT_PATH_SMOTE    = PROCESSED_DIR / "dataset_smote.csv"
METRICS_PATH         = BASE_DIR / "metrics.json"

# ══════════════════════════════════════════════════════════════════════
# 2. COLONNES DU DATASET
# ══════════════════════════════════════════════════════════════════════
COL_PARTICIPANT = "participant"
COL_SESSION     = "session"
COL_TIMESTAMP   = "timestamp"
COL_LABEL       = "label"

# ══════════════════════════════════════════════════════════════════════
# 3. SIGNAUX & LABELS
# ══════════════════════════════════════════════════════════════════════
SIGNAL_COLS = [
    "acc_x", "acc_y", "acc_z",
    "eda", "wrist_hr", "ibi",
    "temp", "breathing_rpm",
    "age", "gender",   # contexte démographique — aide la généralisation inter-participants
]

NUM_COLS = [
    "acc_x", "acc_y", "acc_z",
    "eda", "wrist_hr", "ibi",
    "temp", "breathing_rpm", "breathing_q",
    "age", "gender",
]

LABEL_MAP = {"baseline": -1, "activity": 0, "fatigue": 1}

# ══════════════════════════════════════════════════════════════════════
# 4. HARDWARE STM32 (budget Flash pour TFLite INT8)
# ══════════════════════════════════════════════════════════════════════
STM32_FLASH_KB = 512    # budget conservateur : 2048 KB total - ~1000 KB firmware
STM32_RAM_KB   = 1433   # STM32H7A3ZIT6Q

# ══════════════════════════════════════════════════════════════════════
# 5. OPTUNA — FLAGS & CHEMINS DE RÉSULTATS
# ══════════════════════════════════════════════════════════════════════
USE_OPTUNA_CNN_1D  = False
USE_OPTUNA_LSTM    = True
USE_OPTUNA_LGBM    = False
USE_OPTUNA_CNN_TCN = False
USE_OPTUNA_TCN     = True
USE_OPTUNA_CNN_LSTM = True

OPTUNA_PATH_CNN_D1  = OPTUNA / "optuna_cnn_results.json"
OPTUNA_PATH_LSTM    = OPTUNA / "optuna_lstm_results.json"
OPTUNA_PATH_LGBM    = OPTUNA / "optuna_lgbm_results.json"
OPTUNA_PATH_CNN_TCN = OPTUNA / "optuna_cnn_tcn_results.json"
OPTUNA_PATH_TCN     = OPTUNA / "optuna_tcn_results.json"
OPTUNA_PATH_CNN_LSTM = OPTUNA / "optuna_cnn_lstm_results.json"

# ══════════════════════════════════════════════════════════════════════
# 6. WINDOW CONFIGS — adaptatif par participant (LOSO)
# ══════════════════════════════════════════════════════════════════════
# Participants atypiques : fenêtre élargie + epochs augmentés
# epochs = plafond max (EarlyStopping coupe avant)
WINDOW_CONFIGS = {
    3:  {"window_size": 480, "step_size": 240, "epochs": 500},   # signal atypique
    7:  {"window_size": 480, "step_size": 240, "epochs": 500},   # collapse fatigue
    9:  {"window_size": 480, "step_size": 240, "epochs": 500},   # profil unique
    11: {"window_size": 480, "step_size": 240, "epochs": 500},   # signal atypique
    "default": {"window_size": 240, "step_size": 120, "epochs": 1000},
}

# ══════════════════════════════════════════════════════════════════════
# 7. HYPERPARAMÈTRES PAR DÉFAUT (utilisés si Optuna désactivé)
# ══════════════════════════════════════════════════════════════════════
MODEL_PARAMS = {
    "CNN_1D": {
        "batch_size": 32,
        "n_conv_blocks": 2,
        "kernel_size": 5,
        "filters": 128,
        "use_batchnorm": False,
        "activation": "relu",
        "pool_size": 3,
        "global_pooling": "flatten",
        "l2_reg": 0.0004677279199169856,
        "optimizer": "rmsprop",
        "dense_units": 256,
        "dropout_rate": 0.36790290255054414,
        "learning_rate": 0.00015679008763782025
    },
    "LSTM": {
        "n_lstm_layers": 1,       # 1 ou 2 couches empilées
        "lstm_units":    64,
        "lstm_units_2":  32,      # taille couche 2 si n_lstm_layers=2
        "bidirectional": False,   # True → Bidirectional LSTM
        "dense_units":   64,
        "dropout_rate":  0.4,
        "l2_reg":        1e-4,
        "learning_rate": 0.00001,
        "batch_size":    32,
    },
    "CNN_TCN": {
        "batch_size": 32,
        "cnn_filters": 64,
        "cnn_kernel": 5,
        "n_tcn_blocks": 3,
        "tcn_filters": 64,
        "tcn_kernel": 5,
        "activation": "relu",
        "l2_reg": 1.0407988654029368e-05,
        "optimizer": "rmsprop",
        "dense_units": 64,
        "dropout_rate": 0.30275665469524965,
        "learning_rate": 0.0003734774908783387
    },
    "TCN": {
        "batch_size": 128,
        "n_tcn_blocks": 4,
        "tcn_filters": 64,
        "tcn_kernel": 5,
        "activation": "relu",
        "l2_reg": 1.1554221780153881e-05,
        "optimizer": "rmsprop",
        "dense_units": 32,
        "dropout_rate": 0.3196140167572857,
        "learning_rate": 0.00032811692941705107
    },
    "CNN_LSTM": {
        "batch_size": 32,
        "cnn_filters": 64,
        "cnn_kernel": 5,
        "n_conv_blocks": 2,
        "pool_size": 2,
        "lstm_units": 64,
        "bidirectional": False,
        "activation": "relu",
        "l2_reg": 1e-4,
        "optimizer": "adam",
        "dense_units": 64,
        "dropout_rate": 0.3,
        "learning_rate": 0.0005,
    },
    "LGBM": {
        "objective":         "multiclass",
        "num_class":         3,
        "metric":            "multi_logloss",
        "n_estimators":      500,
        "learning_rate":     0.005,
        "num_leaves":        63,
        "max_depth":         -1,
        "min_child_samples": 20,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "n_jobs":            -1,
        "random_state":      42,
        "verbose":           -1,
    },
}
