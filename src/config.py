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
OUTPUT_PATH_NO_SMOTE = PROCESSED_DIR / "dataset_no_smote.csv"
OUTPUT_PATH_NO_SMOTE_FREQ_NO_SYNC = PROCESSED_DIR / "dataset_no_smote_freq_no_sync.csv"
OUTPUT_PATH_SMOTE = PROCESSED_DIR / "dataset_smote.csv"
METRICS_PATH = BASE_DIR / "metrics.json"
# DATA_MODEL_READY = DATA_DIR / "used" / "dataset.csv"
DATA_MODEL_READY = OUTPUT_PATH_NO_SMOTE

OPTUNA = BASE_DIR / "optuna"
OPTUNA_PATH_CNN_D1 = OPTUNA / "optuna_cnn_results.json"
OPTUNA_PATH_LSTM = OPTUNA / "optuna_lstm_results.json"
OPTUNA_PATH_LGBM = OPTUNA / "optuna_lgbm_results.json"
OPTUNA_PATH_XGB = OPTUNA / "optuna_xgb_results.json"
OPTUNA_PATH_CNN_TCN = OPTUNA / "optuna_cnn_tcn_results.json"
OPTUNA_PATH_TCN = OPTUNA / "optuna_tcn_results.json"
OPTUNA_PATH_CNN_LSTM = OPTUNA / "optuna_cnn_lstm_results.json"
OPTUNA_PATH_ESN = OPTUNA / "optuna_esn_results.json"
OPTUNA_PATH_ESN_RLS = OPTUNA / "optuna_esn_rls_results.json"
OPTUNA_PATH_MESN = OPTUNA / "optuna_mesn_results.json"
OPTUNA_PATH_QRC = OPTUNA / "optuna_qrc_results.json"
OPTUNA_PATH_QLSTM = OPTUNA / "optuna_qlstm_results.json"
OPTUNA_PATH_QNN = OPTUNA / "optuna_qnn_results.json"
OPTUNA_PATH_QKERNEL = OPTUNA / "optuna_qkernel_results.json"
OPTUNA_PATH_DISTILL_STUDENT = OPTUNA / "optuna_distill_student_results.json"
OPTUNA_PATH_DISTILL_TEACHER = OPTUNA / "optuna_distill_teacher_results.json"

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
    "ambient_grandeur.csv": 4,
    "wrist_ibi.csv": 0.65,

    # "features_ppg_ear_right.csv":   100,
    # "ear_gyro_right.csv":         114.86,
}

SIGNAL_COLS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "wrist_hr",
    "ibi",
    "temp",
    "temp_amb",
    "hum_amb",
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
    "wrist_hr.csv": {"hr": "wrist_hr"},
    "wrist_ibi.csv": {"duration": "ibi"},
    "wrist_skin_temperature.csv": {"temp": "temp"},
    "ambient_grandeur.csv":{"temperature_amb":"temp_amb" , "humidite_amb":"hum_amb"},
}

NUM_COLS = SIGNAL_COLS.copy()
NUM_COLS.append("breathing_q")

LABEL_MAP = {"baseline": 0, "activity": 1, "pre_fatigue": 2, "fatigue": 3}
# ══════════════════════════════════════════════════════════════════════
# 4. HARDWARE STM32 (budget Flash pour TFLite INT8)
# ══════════════════════════════════════════════════════════════════════
STM32_FLASH_KB = 512  # budget conservateur : 2048 KB total 
STM32_RAM_KB = 1376  # STM32H7A3ZIT6Q

# ══════════════════════════════════════════════════════════════════════
# 5. OPTUNA — FLAGS & CHEMINS DE RÉSULTATS
# ══════════════════════════════════════════════════════════════════════
USE_OPTUNA_CNN_1D = True
USE_OPTUNA_LSTM = False
USE_OPTUNA_CNN_TCN = False
USE_OPTUNA_TCN = True
USE_OPTUNA_CNN_LSTM = False

USE_OPTUNA_LGBM = True
USE_OPTUNA_XGB = True

USE_OPTUNA_QRC = True
USE_OPTUNA_QLSTM = True
USE_OPTUNA_QNN = False
USE_OPTUNA_QKERNEL = True

USE_OPTUNA_ESN = True
USE_OPTUNA_ESN_RLS = True
USE_OPTUNA_MESN = True

# ══════════════════════════════════════════════════════════════════════
CNN_D1_OPTUNA_TRIALS = 70
CNN_D1_OPTUNA_SESSIONS = 30       

CNN_D1_TCN_OPTUNA_TRIALS = 25
CNN_D1_TCN_OPTUNA_SESSIONS = 36




QRC_OPTUNA_TRIALS = 70
QRC_OPTUNA_SESSIONS = 12

QLSTM_OPTUNA_TRIALS = 20
QLSTM_OPTUNA_SESSIONS = 5

QNN_OPTUNA_TRIALS = 40
QNN_OPTUNA_SESSIONS = 10
QNN_OPTUNA_EPOCHS = 15  # VQC evalue UNE FOIS par fenetre (pas de RNN) -> plus rapide que QLSTM
LOSO_EPOCHS_QNN = 100

QKERNEL_OPTUNA_TRIALS = 100
QKERNEL_OPTUNA_SESSIONS = 20



ESN_OPTUNA_TRIALS = 70
ESN_OPTUNA_SESSIONS = 30

ESN_RLS_OPTUNA_TRIALS = 150
ESN_RLS_OPTUNA_SESSIONS = 30

MESN_OPTUNA_TRIALS = 70
MESN_OPTUNA_SESSIONS = 30



TEACHER_OPTUNA_TRIALS = 40
TEACHER_OPTUNA_SESSIONS = 20
# ══════════════════════════════════════════════════════════════════════
# 6. OPTUNA — ESPACES DE RECHERCHE
# ══════════════════════════════════════════════════════════════════════

LGBM_GPU_PARAMS = {
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "max_bin": 63,  # Requis pour GPU , maximum 255
}


#------------------------------------------------------------------------>>
LSTM_OPTUNA_SPACE = {
    "n_lstm_layers": {"low": 1, "high": 2},
    "lstm_units":    [32, 64, 128],
    "lstm_units_2":  [16, 32, 64],
    "bidirectional": [False, True],
    "dense_units":   [32, 64, 128],
    "dropout_rate":  {"low": 0.2, "high": 0.55},
    "l2_reg":        {"low": 1e-6, "high": 1e-2, "log": True},
    "learning_rate": {"low": 5e-5, "high": 5e-3, "log": True},
    "batch_size":    [16, 32], 
}

CNN_D1_OPTUNA_SPACE = {
    "n_conv_blocks": [1, 2, 3, 4],
    "kernel_size":   [3, 5],
    "filters":       [16,32, 64, 128],
    "use_batchnorm": [True, False],
    "activation":    ["relu", "leaky_relu"],
    "pool_size":     [2, 3],
    "global_pooling":["flatten", "avg", "max"],
    "l2_reg":        (1e-5, 1e-2),
    "optimizer":     ["adam", "rmsprop"],
    "dense_units":   [64, 128, 256],
    "dropout_rate":  (0.2, 0.6),
    "learning_rate": (1e-4, 1e-3),
    "batch_size":    [32, 64, 128],
}

CNN_TCN_OPTUNA_SPACE = {
    "cnn_filters":   [32, 64, 128],
    "cnn_kernel":    [3, 5],
    "n_tcn_blocks":  [2, 3, 4],
    "tcn_filters":   [32, 64, 128],
    "tcn_kernel":    [3, 5],
    "activation":    ["relu", "leaky_relu"],
    "l2_reg":        (1e-5, 1e-2),
    "optimizer":     ["adam", "rmsprop"],
    "dense_units":   [32, 64],
    "dropout_rate":  (0.2, 0.5),
    "learning_rate": (1e-4, 1e-3),
    "batch_size":    [32, 64, 128],
}

CNN_LSTM_OPTUNA_SPACE = {
    "cnn_filters":    [32, 64, 128],
    "cnn_kernel":     [3, 5],
    "n_conv_blocks":  [1, 2, 3],
    "pool_size":      [2, 3],
    "lstm_units":     [32, 64, 128],
    "bidirectional":  [True, False],
    "activation":     ["relu", "leaky_relu"],
    "l2_reg":         (1e-5, 1e-2),
    "optimizer":      ["adam", "rmsprop"],
    "dense_units":    [32, 64, 128],
    "dropout_rate":   (0.2, 0.5),
    "learning_rate":  (1e-4, 1e-3),
    "batch_size":     [32, 64],
}

TCN_OPTUNA_SPACE = {
    "n_tcn_blocks":  [2, 3, 4,5],
    "tcn_filters":   [32, 64,128],
    "tcn_kernel":    [3, 5,7],
    "activation":    ["relu", "leaky_relu"],
    "l2_reg":        (1e-5, 1e-2),
    "optimizer":     ["adam", "rmsprop"],
    "dense_units":   [32, 64,128],
    "dropout_rate":  (0.2, 0.5),
    "learning_rate": (1e-4, 1e-3),
    "batch_size":    [32, 64, 128],
}

#---------------------------------------------------------------------------------------------------------------------------- ML

LGBM_OPTUNA_SPACE = {
    "n_estimators":      (100, 1000),
    "learning_rate":     (1e-3, 0.3),
    "num_leaves":        (15, 127),
    "max_depth":         [-1, 6, 10, 15],
    "min_child_samples": (5, 50),
    "subsample":         (0.5, 1.0),
    "colsample_bytree":  (0.5, 1.0),
    "reg_alpha":         (1e-4, 10.0),
    "reg_lambda":        (1e-4, 10.0),
    **LGBM_GPU_PARAMS,
}


XGB_OPTUNA_SPACE = {
    "n_estimators":      (50, 500),
    "learning_rate":     (0.01, 0.3),
    "max_depth":         (3, 12),
    "min_child_weight":  (1, 10),
    "subsample":         (0.6, 1.0),
    "colsample_bytree":  (0.6, 1.0),
    "reg_alpha":         (1e-8, 10.0),
    "reg_lambda":        (1e-8, 10.0),
    "gamma":             (0.0, 5.0),
}


#---------------------------------------------------------------------------------------------------------------------------- Quantum

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

QRC_OPTUNA_SPACE = {
    "n_qubits": [5, 6, 7, 8],
    "n_layers": [2, 3, 4],
    "input_scaling": {"low": 0.05, "high": 1.0, "log": True},
    "reservoir_scale": {"low": 0.2, "high": 2.0, "log": True},
    "feedback_scale": {"low": 0.0, "high": 1.5},
    "leak_rate": {"low": 0.1, "high": 0.7},
    "ridge_alpha": {"low": 1e-3, "high": 50.0, "log": True},
    "temporal_stride": [4, 6, 8, 12],  # ← Retiré 2 (trop lent, 96 pas)
    "state_summary": ["last_mean", "last_mean_std"],
}

QNN_OPTUNA_SPACE = {
    "n_qubits": [4, 5, 6, 7, 8],
    "n_vqc_layers": [1, 2, 3],
    "dense_units": [32, 64, 128],
    "dropout_rate": {"low": 0.1, "high": 0.5},
    "l2_reg": {"low": 1e-6, "high": 1e-2, "log": True},
    "learning_rate": {"low": 1e-4, "high": 1e-2, "log": True},
    "batch_size": [16, 32],
}

QKERNEL_OPTUNA_SPACE = {
    "n_qubits": [4, 5, 6, 7, 8, 9, 10],
    "input_scaling": {"low": 0.001, "high": 1.0, "log": True},
    "svm_C": {"low": 0.001, "high": 30.0, "log": True},
}

#----------------------------------------------------------------------------------------------------------------------------

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
            "ambient": [20, 40, 80],
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
        "n_conv_blocks": 3,
        "kernel_size": 5,
        "filters": 32,
        "use_batchnorm": True,
        "activation": "relu",
        "pool_size": 3,
        "global_pooling": "max",
        "l2_reg": 0.004384004019043003,
        "optimizer": "adam",
        "dense_units": 256,
        "dropout_rate": 0.41303959172498744,
        "learning_rate": 0.0008280917627432003
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
    "LSTM de optuna dernier run": {
            "n_lstm_layers": 1,
            "lstm_units": 32,
            "bidirectional": True,
            "dense_units": 128,
            "dropout_rate": 0.40111054193208334,
            "l2_reg": 0.006873016554617295,
            "learning_rate": 0.002318363915331103,
            "batch_size": 32
    },
    "CNN_TCN": {
        "batch_size": 32,
        "cnn_filters": 128,
        "cnn_kernel": 5,
        "n_tcn_blocks": 4,
        "tcn_filters": 64,
        "tcn_kernel": 5,
        "activation": "relu",
        "l2_reg": 3.612012051387347e-05,
        "optimizer": "adam",
        "dense_units": 32,
        "dropout_rate": 0.39200912336963123,
        "learning_rate": 0.00039184636569582513
    },
    "CNN_LSTM":{
        "batch_size": 32,
        "cnn_filters": 64,
        "cnn_kernel": 3,
        "n_conv_blocks": 3,
        "pool_size": 3,
        "lstm_units": 128,
        "bidirectional": True,
        "activation": "leaky_relu",
        "l2_reg": 0.00347312383864753,
        "optimizer": "adam",
        "dense_units": 32,
        "dropout_rate": 0.22700687336676065,
        "learning_rate": 0.000748237189143471
    },
    "TCN": {
        "batch_size": 64,
        "n_tcn_blocks": 3,
        "tcn_filters": 128,
        "tcn_kernel": 3,
        "activation": "leaky_relu",
        "l2_reg": 0.0029753822912700384,
        "optimizer": "rmsprop",
        "dense_units": 32,
        "dropout_rate": 0.4486277698278758,
        "learning_rate": 0.0007720563684349782
    },
    "LGBM": {
        "n_estimators": 437,
        "learning_rate": 0.22648248189516848,
        "num_leaves": 97,
        "max_depth": -1,
        "min_child_samples": 44,
        "subsample": 0.8005575058716043,
        "colsample_bytree": 0.8540362888980227,
        "reg_alpha": 0.00012674255898937226,
        "reg_lambda": 7.072114131472227
    },
    "XGB": {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.0,
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
    "QNN": {
        "n_qubits": 6,
        "n_vqc_layers": 2,
        "dense_units": 32,
        "dropout_rate": 0.34851616147340775,
        "l2_reg": 3.378104721677979e-06,
        "learning_rate": 0.0059619848416953836,
        "batch_size": 32,
    },
    "QKERNEL": {
        "n_qubits": 6,
        "input_scaling": 0.5,
        "svm_C": 1.0,
        "random_state": 42,
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
    "DISTILL_TEACHER": {
        "patch_size": 16,
        "d_model": 64,
        "num_heads": 4,
        "d_ff": 256,
        "num_layers": 4,
        "dropout": 0.1,
        "dense_units": 64,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
    },
    "DISTILL_STUDENT": {
        "batch_size": 32,
        "n_conv_blocks": 2,
        "kernel_size": 3,
        "filters": 32,
        "use_batchnorm": True,
        "activation": "relu",
        "pool_size": 2,
        "global_pooling": "avg",
        "l2_reg": 1e-4,
        "optimizer": "adam",
        "dense_units": 64,
        "dropout_rate": 0.3,
        "learning_rate": 5e-4,
        "alpha": 0.5,
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

    "MESN" : {
        "ridge_alpha": 0.5617197644275693,
        "washout": 0,
        "state_summary": "last_mean_std",
        "min_coverage": 0.3532539281450102,
        "random_state": 42,
        "reservoirs": {
            "acc": {
                "n_reservoir": 200,
                "spectral_radius": 0.8868691168996753,
                "sparsity": 0.9654014708429202,
                "leak_rate": 0.7185960594010997,
                "input_scaling": 0.20599342543830018
            },
            "ambient": {
                "n_reservoir": 80,
                "spectral_radius": 1.1832418542857157,
                "sparsity": 0.9639639007570672,
                "leak_rate": 0.39576016255301916,
                "input_scaling": 0.7019561164367488
            },
            "hr": {
                "n_reservoir": 60,
                "spectral_radius": 0.9554788965712058,
                "sparsity": 0.8878071135425127,
                "leak_rate": 0.33159232923633486,
                "input_scaling": 0.056810378023327494
            },
            "ibi": {
                "n_reservoir": 20,
                "spectral_radius": 1.1980179295026065,
                "sparsity": 0.9473013259954898,
                "leak_rate": 0.4715080087104783,
                "input_scaling": 0.6299332968311588
            },
            "temp": {
                "n_reservoir": 40,
                "spectral_radius": 0.3757945542965706,
                "sparsity": 0.8760651524610491,
                "leak_rate": 0.596043139415017,
                "input_scaling": 1.3927337457057325
            },
            "breathing": {
                "n_reservoir": 70,
                "spectral_radius": 0.6567098047512803,
                "sparsity": 0.881578233604194,
                "leak_rate": 0.6736843364131624,
                "input_scaling": 0.09011997063968813
            }
        }
    }
}
