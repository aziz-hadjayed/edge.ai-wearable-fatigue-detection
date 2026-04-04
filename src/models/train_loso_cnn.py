import json
import os
import sys

# Ajouter src/ au path pour trouver config.py
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# choix ressource
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # pour utiliser le cpu

#
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# Config
from config import *

optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------------------------------------------------------------------------------------//params





# Categorical mapping: to use keras to_categorical, labels must be 0, 1, 2
LABEL_MAPPING = {-1: 0, 0: 1, 1: 2}
TARGET_NAMES = ["baseline (-1)", "activity (0)", "fatigue (1)"]
MODEL_NAME = "CNN_1D_LOSO"


# ----------------------------------------------------------------------------------------------//functions
def extract_windows_from_group(group_df, window_size, step_size):
    """
    Extract overlapping continuous windows from a single session.
    No timestamp used in features. Label is the majority vote.
    """
    X_group = group_df[SIGNAL_COLS].values
    y_group = group_df[COL_LABEL].values

    windows_x = []
    windows_y = []

    n_samples = len(X_group)
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        window_x = X_group[start:end]
        window_y_raw = y_group[start:end]

        # Majority vote for label
        vals, counts = np.unique(window_y_raw, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_label = vals[majority_idx]

        windows_x.append(window_x)
        windows_y.append(majority_label)

    return windows_x, windows_y


def extract_all_windows(df, window_size, step_size):
    """
    Group by participant and session to ensure NO windows overlap across sessions.
    """
    X_all = []
    y_all = []

    # Sort optionally by participant, session, timestamp to be safe
    if "timestamp" in df.columns:
        df = df.sort_values([COL_PARTICIPANT, COL_SESSION, "timestamp"])

    for (part, sess), group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        wx, wy = extract_windows_from_group(group, window_size, step_size)
        X_all.extend(wx)
        y_all.extend(wy)

    return np.array(X_all), np.array(y_all)


def build_cnn_1d(input_shape, num_classes):
    """
    Simple 1D CNN Architecture.
    """
    model = Sequential(
        [
            Conv1D(filters=16, kernel_size=3, activation="relu", input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(100, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_cnn_1d_optuna(trial, input_shape, num_classes):
    """
    CNN 1D dont l'architecture est définie par un trial Optuna.
    Hyperparamètres recherchés :
      - n_conv_blocks  : nombre de blocs Conv1D (1 à 3)
      - filters_i      : nombre de filtres par bloc
      - kernel_size    : taille du noyau de convolution
      - use_batchnorm  : BatchNormalization après chaque Conv
      - dense_units    : neurones de la couche Dense finale
      - dropout_rate   : taux de Dropout
      - learning_rate  : taux d'apprentissage Adam
    """
    n_conv_blocks = trial.suggest_int("n_conv_blocks", 1, 3)
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
    dense_units = trial.suggest_categorical("dense_units", [64, 128, 256])
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    layers = []
    for i in range(n_conv_blocks):
        filters = trial.suggest_categorical(f"filters_{i}", [16, 32, 64, 128])
        if i == 0:
            layers.append(
                Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", input_shape=input_shape)
            )
        else:
            layers.append(Conv1D(filters=filters, kernel_size=kernel_size, activation="relu"))
        if use_batchnorm:
            layers.append(BatchNormalization())
        # Pooling uniquement si la dimension temporelle reste suffisante
        if input_shape[0] // (2 ** (i + 1)) > 1:
            layers.append(MaxPooling1D(pool_size=2))

    layers += [
        Flatten(),
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax"),
    ]

    model = Sequential(layers)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def optuna_objective(trial, df, num_classes, n_optuna_epochs=5, n_val_folds=3):
    """
    Objectif Optuna : évalue un trial sur n_val_folds participants distincts
    et retourne la moyenne des F1-Macro — plus robuste qu'un seul fold.

    Pour chaque fold :
      - train  = tous les participants sauf le participant de validation
      - val    = 1 participant (rotation sur les n_val_folds premiers)
    """
    config = WINDOW_CONFIGS["default"]
    window_size = config["window_size"]
    step_size = config["step_size"]
    input_shape = (window_size, len(SIGNAL_COLS))

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    participants = df[COL_PARTICIPANT].unique()
    val_participants = participants[:n_val_folds]

    scores = []
    for fold_idx, val_part in enumerate(val_participants):
        df_train = df[df[COL_PARTICIPANT] != val_part].copy()
        df_val = df[df[COL_PARTICIPANT] == val_part].copy()

        scaler = RobustScaler()
        df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
        df_val[SIGNAL_COLS] = scaler.transform(df_val[SIGNAL_COLS])

        X_train, y_train = extract_all_windows(df_train, window_size, step_size)
        X_val, y_val = extract_all_windows(df_val, window_size, step_size)

        if len(X_val) == 0:
            continue

        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = {cls: w for cls, w in zip(classes, weights)}

        y_train_cat = to_categorical(y_train, num_classes=num_classes)

        tf.keras.backend.clear_session()
        model = build_cnn_1d_optuna(trial, input_shape, num_classes)
        model.fit(
            X_train,
            y_train_cat,
            epochs=n_optuna_epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            verbose=0,
        )

        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        fold_score = f1_score(y_val, y_pred, average="macro", zero_division=0)
        scores.append(fold_score)

        # Pruning intermédiaire : abandonne ce trial si les premiers folds sont mauvais
        trial.report(fold_score, step=fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if len(scores) == 0:
        raise optuna.exceptions.TrialPruned()

    return float(np.mean(scores))


def optimize_hyperparams(df, num_classes, n_trials=30, n_optuna_epochs=5, n_val_folds=3):
    """
    Lance une étude Optuna pour trouver les meilleurs hyperparamètres CNN.
    Évalue chaque trial sur n_val_folds participants (moyenne des F1-Macro).
    Retourne le dictionnaire des meilleurs paramètres.
    """
    print(f"\n{'='*60}")
    print(f"OPTUNA — Recherche d'hyperparamètres")
    print(f"  trials      : {n_trials}")
    print(f"  folds/trial : {n_val_folds}  (×{n_optuna_epochs} epochs chacun)")
    print(f"  objectif    : F1-Macro moyen sur {n_val_folds} participants")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, df, num_classes, n_optuna_epochs, n_val_folds),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nMeilleur F1-Macro moyen ({n_val_folds} folds) : {study.best_value:.4f}")
    print(f"Meilleurs hyperparamètres :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Importance des hyperparamètres (dont n_conv_blocks)
    print(f"\nImportance des hyperparamètres :")
    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in importances.items():
            bar = "█" * int(importance * 40)
            print(f"  {param:<20} {bar} {importance:.3f}")
    except Exception:
        print("  (pas assez de trials pour calculer l'importance)")

    # Sauvegarde des résultats Optuna
    optuna_results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "n_val_folds": n_val_folds,
        "n_optuna_epochs": n_optuna_epochs,
    }
    OPTUNA_PATH_CNN_D1.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_CNN_D1, "w") as f:
        json.dump(optuna_results, f, indent=4)
    print(f"\nRésultats Optuna sauvegardés : {OPTUNA_PATH_CNN_D1}")

    return study.best_params

# ______________________________________________________________________________________________//main
def main():

    print("\n" + "=" * 60)
    print(f"TRAIN LOSO CNN")
    print("=" * 60)

    # Load dataset____________________________________________________________
    if not DATA_PROCESSED.exists():
        print(f"Dataset not found: {DATA_PROCESSED}")
        print("Please ensure `main.py` successfully generated the processed dataset.")
        return

    print(f"Loading dataset from {DATA_PROCESSED}")
    df = pd.read_csv(DATA_PROCESSED)
    # ________________________________________________________________________

    # Map the labels to categorical 0, 1, 2 et afficher les participants________
    df[COL_LABEL] = df[COL_LABEL].map(LABEL_MAPPING)

    participants = df[COL_PARTICIPANT].unique()
    print(f"Found {len(participants)} participants: {participants}")
    # ________________________________________________________________________

    num_classes = len(LABEL_MAPPING)
    all_metrics = []

    # choisir si on veut utiliser optuna ou les hyperparamètres par défaut____
    if USE_OPTUNA_CNN_1D:
        best_params = optimize_hyperparams(df, num_classes, n_trials=30, n_optuna_epochs=5)
    else:
        best_params = MODEL_PARAMS["CNN_1D"]
        print("\nOptuna désactivé — hyperparamètres par défaut utilisés.")
    # ________________________________________________________________________

    # LOSO cross-validation_______________________________________________________________________________
    #_____________________________________________________________________________________________________
    for test_idx, test_part in enumerate(participants):
        print(f"\n" + "=" * 60)
        print(
            f"FOLD {test_idx + 1}/{len(participants)} — TEST PARTICIPANT: {test_part}"
        )
        print("=" * 60)

        # Sélectionner configuration adaptée au participant____________________
        if test_part in WINDOW_CONFIGS:
            config = WINDOW_CONFIGS[test_part]
            print(f"CONFIG ADAPTÉE (participant difficile)")
        else:
            config = WINDOW_CONFIGS["default"]
            print(f"CONFIG STANDARD")

        # Appliquer configuration pour ce fold
        WINDOW_SIZE = config["window_size"]
        STEP_SIZE = config["step_size"]
        EPOCHS = config["epochs"]
        print(f"Window: {WINDOW_SIZE} | Step: {STEP_SIZE} | Epochs: {EPOCHS}")
        # ________________________________________________________________________

        # LOSO Split
        df_train = df[df[COL_PARTICIPANT] != test_part].copy()
        df_test = df[df[COL_PARTICIPANT] == test_part].copy()

        # Normalisation par fold : fit sur train uniquement → pas de leakage
        scaler = RobustScaler()
        df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
        df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])

        print("Extracting sliding windows...")
        X_train, y_train = extract_all_windows(df_train, WINDOW_SIZE, STEP_SIZE)
        X_test, y_test = extract_all_windows(df_test, WINDOW_SIZE, STEP_SIZE)

        print(f"Train windows: {len(X_train)} | Test windows: {len(X_test)}")

        if len(X_test) == 0:
            print(
                f"WARNING: Test set for participant {test_part} is empty. Skipping fold."
            )
            continue

        # Calculate balanced class weights based ONLY on training set
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
        print(f"Class weights (Train): {class_weight_dict}")

        # Convert targets to one-hot encoding for categorical crossentropy
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        # Note: we keep y_test simple for evaluation, but can be categorical if used in evaluate
        # y_test_cat = to_categorical(y_test, num_classes=num_classes)

        # Build Model avec les meilleurs hyperparamètres Optuna
        input_shape = (WINDOW_SIZE, len(SIGNAL_COLS))
        tf.keras.backend.clear_session()
        fixed_trial = optuna.trial.FixedTrial(best_params)
        model = build_cnn_1d_optuna(fixed_trial, input_shape=input_shape, num_classes=num_classes)

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True)
        # Train
        print(f"Training model on {len(X_train)} samples across {EPOCHS} epochs...")
        model.fit(
            X_train,
            y_train_cat,
            epochs=EPOCHS,
            validation_split=0.1,  
            batch_size=BATCH_SIZE_CNN_1D,
            callbacks= early_stop,
            class_weight=class_weight_dict,
            verbose=1,
        )

        # Test Evaluation
        print(
            f"Evaluating model on {len(X_test)} samples for Participant {test_part}..."
        )
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Ensure all shapes match
        f1_mac = f1_score(y_test, y_pred, average="macro")
        f1_wei = f1_score(y_test, y_pred, average="weighted")
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        print(f"\n>>> Fold {test_idx + 1} Metrics:")
        print(f"F1-Macro:          {f1_mac:.4f}")
        print(f"F1-Weighted:       {f1_wei:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")

        report = classification_report(
            y_test,
            y_pred,
            target_names=TARGET_NAMES[: len(np.unique(np.concatenate([y_test, y_pred])))],
            labels=np.unique(np.concatenate([y_test, y_pred])),
            zero_division=0,
        )
        print("\nClassification Report:\n", report)

        # ---------------------------- Save model ---------------------------------
        model_stem = f"{MODEL_NAME}_fold{test_idx + 1}_testP{test_part}"
        save_dir = MODELS_DIR / "CNN-1D"
        save_dir.mkdir(parents=True, exist_ok=True)

        # .tflite INT8 — TFLite Micro (STM32 / MCU)________________________________
        def representative_dataset():
            sample = X_train[:200].astype(np.float32)
            for i in range(len(sample)):
                yield [sample[i : i + 1]]

        converter_micro = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_micro.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_micro.representative_dataset = representative_dataset
        converter_micro.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_micro.inference_input_type = tf.int8
        converter_micro.inference_output_type = tf.int8
        tflite_micro_model = converter_micro.convert()

        tflite_micro_path = save_dir / f"{model_stem}_int8.tflite"
        tflite_micro_path.write_bytes(tflite_micro_model)
        print(f"✔ Modèle TFLite Micro INT8 sauvegardé: {tflite_micro_path}")
        # __________________________________________________________________________

        # C array (.h) pour TFLite Micro____________________________________________
        c_array_path = save_dir / f"{model_stem}_int8.h"
        var_name = model_stem.replace("-", "_").replace(".", "_")
        hex_bytes = ", ".join(f"0x{b:02x}" for b in tflite_micro_model)
        c_header = (
            f"// Auto-generated TFLite Micro model — {model_stem}\n"
            f"// Window: {WINDOW_SIZE} | Features: {len(SIGNAL_COLS)} | Classes: {num_classes}\n\n"
            f"#ifndef {var_name.upper()}_H\n"
            f"#define {var_name.upper()}_H\n\n"
            f"#include <stdint.h>\n\n"
            f"const unsigned int {var_name}_len = {len(tflite_micro_model)};\n"
            f"alignas(8) const uint8_t {var_name}[] = {{\n  {hex_bytes}\n}};\n\n"
            f"#endif  // {var_name.upper()}_H\n"
        )
        c_array_path.write_text(c_header)
        print(f"✔ C array TFLite Micro sauvegardé    : {c_array_path}")
        # __________________________________________________________________________





        all_metrics.append(
            {
                "Fold": test_idx + 1,
                "Test_Participant": test_part,
                "F1_Macro": f1_mac,
                "F1_Weighted": f1_wei,
                "Balanced_Accuracy": bal_acc,
                "Window_Size": WINDOW_SIZE,
                "Step_Size": STEP_SIZE,
                "Epochs": EPOCHS,
                "Config_Type": "ADAPTIVE"
                if test_part in WINDOW_CONFIGS
                else "STANDARD",
            }
        )
    #_____________________________________________________________________________________________________
    #_____________________________________________________________________________________________________



    # Overall Summary terminal ___________________________________________________________
    print("\n" + "=" * 60)
    print("GLOBAL EXPERIMENT RESULTS")
    print("=" * 60)

    if len(all_metrics) == 0:
        print("No valid folds completed. Exiting.")
        return

    df_results = pd.DataFrame(all_metrics)

    mean_row = {
        "Fold": "MEAN",
        "Test_Participant": "ALL",
        "F1_Macro": df_results["F1_Macro"].mean(),
        "F1_Weighted": df_results["F1_Weighted"].mean(),
        "Balanced_Accuracy": df_results["Balanced_Accuracy"].mean(),
        "Window_Size": "-",
        "Step_Size": "-",
        "Epochs": "-",
        "Config_Type": "-",
    }

    std_row = {
        "Fold": "STD",
        "Test_Participant": "ALL",
        "F1_Macro": df_results["F1_Macro"].std(),
        "F1_Weighted": df_results["F1_Weighted"].std(),
        "Balanced_Accuracy": df_results["Balanced_Accuracy"].std(),
        "Window_Size": "-",
        "Step_Size": "-",
        "Epochs": "-",
        "Config_Type": "-",
    }

    df_results = pd.concat(
        [df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True
    )
    print("\nSummary metrics per fold over all tests:")
    print(df_results.tail(2).to_string(index=False))
    # __________________________________________________________________________

    # Save JSON_________________________________________________________________
    metrics_json = {
        "model_name": MODEL_NAME,
        "training_strategy": "LOSO_ADAPTIVE",
        "adaptive_config": {
            "description": "Participants difficiles reçoivent fenêtres + epochs augmentés",
            "difficult_participants": [3, 7, 11],
            "configs": WINDOW_CONFIGS,
        },
        "hyperparameters": {
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "epochs": EPOCHS,
            "batch_size": best_params.get("batch_size", BATCH_SIZE_CNN_1D),
            "optuna_best_params": best_params,
        },
        "folds": [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.floating, float))
                    else int(v)
                    if isinstance(v, (np.integer, int))
                    else v
                )
                for k, v in m.items()
            }
            for m in all_metrics
        ],
        "summary": {
            "F1_Macro_mean": float(mean_row["F1_Macro"]),
            "F1_Macro_std": float(std_row["F1_Macro"]),
            "F1_Weighted_mean": float(mean_row["F1_Weighted"]),
            "F1_Weighted_std": float(std_row["F1_Weighted"]),
            "Balanced_Accuracy_mean": float(mean_row["Balanced_Accuracy"]),
            "Balanced_Accuracy_std": float(std_row["Balanced_Accuracy"]),
        },
    }
    json_path = BASE_DIR / "metrics.json"
    all_models_metrics = {}
    if json_path.exists():
        with open(json_path, "r") as f:
            all_models_metrics = json.load(f)
    all_models_metrics[MODEL_NAME] = metrics_json
    with open(json_path, "w") as f:
        json.dump(all_models_metrics, f, indent=4)

    print(f"Metrics JSON exported to    {json_path}")
    # __________________________________________________________________________


if __name__ == "__main__":
    # Disable eager execution spam/memory if needed
    # tf.get_logger().setLevel('ERROR')
    main()
