from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# 1. CHEMINS
# ══════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "01_raw"
INTERIM_DIR = DATA_DIR / "02_interim"
PROCESSED_DIR = DATA_DIR / "03_processed"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models_saved"
DATA_RAW = RAW_DIR
DATA_PROCESSED = PROCESSED_DIR / "dataset_ref.csv"
METADATA_PATH = RAW_DIR / "metadata.csv"
DATA_MODEL_READY = DATA_DIR / "used" / "dataset.csv"
OUTPUT_PATH_NO_SMOTE = PROCESSED_DIR / "dataset_no_smote.csv"
OUTPUT_PATH_SMOTE = PROCESSED_DIR / "dataset_smote.csv"
METRICS_PATH = BASE_DIR / "metrics.json"

OPTUNA = BASE_DIR / "optuna"
OPTUNA_PATH_CNN_D1 = OPTUNA / "optuna_cnn_results.json"
OPTUNA_PATH_LSTM = OPTUNA / "optuna_lstm_results.json"
OPTUNA_PATH_LGBM = OPTUNA / "optuna_lgbm_results.json"
OPTUNA_PATH_CNN_TCN = OPTUNA / "optuna_cnn_tcn_results.json"
OPTUNA_PATH_TCN = OPTUNA / "optuna_tcn_results.json"
OPTUNA_PATH_CNN_LSTM = OPTUNA / "optuna_cnn_lstm_results.json"
OPTUNA_PATH_ESN = OPTUNA / "optuna_esn_results.json"
OPTUNA_PATH_ESN_RLS = OPTUNA / "optuna_esn_rls_results.json"
OPTUNA_PATH_MESN = OPTUNA / "optuna_mesn_results.json"
OPTUNA_PATH_QRC = OPTUNA / "optuna_qrc_results.json"
OPTUNA_PATH_QLSTM = OPTUNA / "optuna_qlstm_results.json"

# ══════════════════════════════════════════════════════════════════════
# 2. COLONNES DU DATASET PRINCIPAL (fusionné & étiqueté)
# ══════════════════════════════════════════════════════════════════════
COL_PARTICIPANT = "participant"
COL_SESSION = "session"
COL_TIMESTAMP = "timestamp"
COL_LABEL = "label"

# ══════════════════════════════════════════════════════════════════════
# 3. SIGNAUX & LABELS
# ══════════════════════════════════════════════════════════════════════
SENSOR_FREQ = {
    "wrist_acc.csv": 32,
    "wrist_hr.csv": 1,
    "wrist_skin_temperature.csv": 4,
    "ambient_grandeur.csv": 100,
    "wrist_ibi.csv": 0.65,
    # "features_ppg_ear_right.csv":   100,
    # "ear_gyro_right.csv":         114.86,
}

SIGNAL_COLS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "eda",
    "wrist_hr",
    "ibi",
    "temp",
    "breathing_rpm",
    "age",
    "gender",
    "reserved",
    "generally_trusting",
    "lazy",
    "relaxed_handless_stress",
    "few_artistic_interests",
    "sociable",
    "criticize_others",
    "thorough_job",
    "easily_nervous",
    "active_magination",
    "phisical_fitness",
    "cardiorespiratory_fitness",
    "muscular_strength",
    "agility_speed",
    "flexibility",
    "cardiovascular_health_conditions",  # contexte démographique — aide la généralisation inter-participants
]

FILE_COLS = {
    "wrist_acc.csv": {"ax": "acc_x", "ay": "acc_y", "az": "acc_z"},
    "wrist_eda.csv": {"eda": "eda"},
    "wrist_hr.csv": {"hr": "wrist_hr"},
    "wrist_ibi.csv": {"duration": "ibi"},
    "wrist_skin_temperature.csv": {"temp": "temp"},
}

NUM_COLS = SIGNAL_COLS.copy()
NUM_COLS.append("breathing_q")

LABEL_MAP = {"baseline": 0, "activity": 1, "pre_fatigue": 2, "fatigue": 3}
# ══════════════════════════════════════════════════════════════════════
# 4. HARDWARE STM32 (budget Flash pour TFLite INT8)
# ══════════════════════════════════════════════════════════════════════
STM32_FLASH_KB = 512  # budget conservateur : 2048 KB total - ~1000 KB firmware
STM32_RAM_KB = 1433  # STM32H7A3ZIT6Q

# ══════════════════════════════════════════════════════════════════════
# 5. OPTUNA — FLAGS & CHEMINS DE RÉSULTATS
# ══════════════════════════════════════════════════════════════════════
USE_OPTUNA_CNN_1D = False
USE_OPTUNA_LSTM = False
USE_OPTUNA_LGBM = True
USE_OPTUNA_CNN_TCN = False
USE_OPTUNA_TCN = False
USE_OPTUNA_CNN_LSTM = True
USE_OPTUNA_ESN = True
USE_OPTUNA_ESN_RLS = True
USE_OPTUNA_MESN = True
USE_OPTUNA_QRC = True
USE_OPTUNA_QLSTM = False


ESN_OPTUNA_TRIALS = 70
ESN_OPTUNA_SESSIONS = 36
ESN_RLS_OPTUNA_TRIALS = 150
ESN_RLS_OPTUNA_SESSIONS = 36
MESN_OPTUNA_TRIALS = 20
MESN_OPTUNA_SESSIONS = 12
QRC_OPTUNA_TRIALS = 70
QRC_OPTUNA_SESSIONS = 12
QLSTM_OPTUNA_TRIALS = 20
QLSTM_OPTUNA_SESSIONS = 5


# ══════════════════════════════════════════════════════════════════════
# 6. OPTUNA — ESPACES DE RECHERCHE
# ══════════════════════════════════════════════════════════════════════

QLSTM_OPTUNA_SPACE = {
    "n_qubits": [2, 4],  # qubits par VQC — plus = plus expressif mais plus lent
    "n_vqc_layers": [1, 2],  # profondeur StronglyEntanglingLayers
    "n_qlstm_layers": [1, 2],  # cellules QLSTM empilées
    "dense_units": [32, 64],
    "dropout_rate": {"low": 0.1, "high": 0.5},
    "l2_reg": {"low": 1e-6, "high": 1e-2, "log": True},
    "learning_rate": {"low": 5e-4, "high": 5e-3, "log": True},
    "batch_size": [8, 16],
}

ESN_OPTUNA_SPACE = {
    "n_reservoir": [100, 150, 200, 300, 400],
    "spectral_radius": {"low": 0.2, "high": 1.4},
    "sparsity": {"low": 0.80, "high": 0.98},
    "leak_rate": {"low": 0.05, "high": 1.0},
    "input_scaling": {"low": 0.05, "high": 1.5, "log": True},
    "ridge_alpha": {"low": 1e-4, "high": 100.0, "log": True},
    "washout": [0, 10, 20, 40],
    "state_summary": ["last", "last_mean", "last_mean_std"],
}

ESN_RLS_OPTUNA_SPACE = {
    "n_reservoir": [100, 150, 200, 300, 400],
    "spectral_radius": {"low": 0.2, "high": 1.4},
    "sparsity": {"low": 0.80, "high": 0.98},
    "leak_rate": {"low": 0.05, "high": 1.0},
    "input_scaling": {"low": 0.05, "high": 1.5, "log": True},
    "rls_alpha": {"low": 1e-2, "high": 10.0, "log": True},
    "forgetting": {"low": 0.985, "high": 1.0},
    "washout": [0, 10, 20, 40],
    "state_summary": ["last", "last_mean", "last_mean_std"],
}

MESN_OPTUNA_SPACE = {
    "ridge_alpha": {"low": 1e-2, "high": 100.0, "log": True},
    "washout": [0, 5, 10, 20],
    "state_summary": ["last", "last_mean", "last_mean_std"],
    "min_coverage": {"low": 0.20, "high": 0.60},
    "reservoirs": {
        "n_reservoir": {
            "acc": [40, 80, 120, 200],
            "eda": [20, 40, 80],
            "hr": [20, 40, 60],
            "ibi": [20, 40, 60],
            "temp": [20, 40, 60],
            "breathing": [40, 70, 100, 150],
        },
        "spectral_radius": {"low": 0.3, "high": 1.3},
        "sparsity": {"low": 0.80, "high": 0.98},
        "leak_rate": {"low": 0.05, "high": 0.80},
        "input_scaling": {"low": 0.05, "high": 1.5, "log": True},
    },
}

QRC_OPTUNA_SPACE = {
    "n_qubits": [5, 6, 7, 8],
    "n_layers": [2, 3, 4],
    "input_scaling": {"low": 0.3, "high": 3.0, "log": True},
    "reservoir_scale": {"low": 0.2, "high": 2.0, "log": True},
    "feedback_scale": {"low": 0.0, "high": 1.5},
    "leak_rate": {"low": 0.1, "high": 0.7},
    "ridge_alpha": {"low": 1e-3, "high": 50.0, "log": True},
    "temporal_stride": [4, 6, 8, 12],  # ← Retiré 2 (trop lent, 96 pas)
    "state_summary": ["last_mean", "last_mean_std"],
}


# ══════════════════════════════════════════════════════════════════════
# 7. WINDOW CONFIGS — adaptatif par participant (LOSO)
# ══════════════════════════════════════════════════════════════════════
# Participants atypiques : fenêtre élargie + epochs augmentés
# epochs = plafond max (EarlyStopping coupe avant)
WINDOW_CONFIGS = {
    1000: {"window_size": 480, "step_size": 240, "epochs": 500},  # signal atypique
    "default": {"window_size": 240, "step_size": 120, "epochs": 1000},
}

# ══════════════════════════════════════════════════════════════════════
# 8. HYPERPARAMÈTRES PAR DÉFAUT (utilisés si Optuna désactivé)
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
        "learning_rate": 0.00015679008763782025,
    },
    "LSTM": {
        "n_lstm_layers": 1,
        "lstm_units": 128,
        "lstm_units_2": 32,  # taille couche 2 si n_lstm_layers=2
        "bidirectional": True,
        "dense_units": 32,
        "dropout_rate": 0.3223855899459306,
        "l2_reg": 0.009928048379348684,
        "learning_rate": 0.00044691892803364514,
        "batch_size": 32,
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
        "learning_rate": 0.0003734774908783387,
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
        "learning_rate": 0.00032811692941705107,
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
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "n_estimators": 113,
        "learning_rate": 0.04230441275950031,
        "num_leaves": 65,
        "max_depth": 15,
        "min_child_samples": 16,
        "subsample": 0.9599864108985356,
        "colsample_bytree": 0.6738167152482913,
        "reg_alpha": 0.04566190143949176,
        "reg_lambda": 0.5002488288570663,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    },
    "QLSTM": {
        "n_qubits": 2,
        "n_vqc_layers": 1,
        "n_qlstm_layers": 1,
        "dense_units": 64,
        "dropout_rate": 0.3,
        "l2_reg": 1e-5,
        "learning_rate": 1e-3,
        "batch_size": 16,
    },
    "ESN": {
        "n_reservoir": 200,
        "spectral_radius": 0.5426589509604672,
        "sparsity": 0.8744991090833628,
        "leak_rate": 0.3657390633899312,
        "input_scaling": 0.289188262765513,
        "ridge_alpha": 5.597203833019878,
        "washout": 10,
        "state_summary": "last_mean_std",
        "random_state": 42,
    },
    "ESN_RLS": {
        "n_reservoir": 200,
        "spectral_radius": 0.9,
        "sparsity": 0.90,
        "leak_rate": 0.3,
        "input_scaling": 0.5,
        "rls_alpha": 0.1,
        "forgetting": 0.995,
        "washout": 20,
        "state_summary": "last_mean_std",
        "random_state": 42,
    },
    "QRC": {
        "n_qubits": 3,
        "n_layers": 1,
        "input_scaling": 0.4,
        "reservoir_scale": 0.6,
        "feedback_scale": 0.25,
        "leak_rate": 0.5,
        "ridge_alpha": 1.0,
        "temporal_stride": 8,
        "state_summary": "last_mean_std",
        "random_state": 42,
    },
}
