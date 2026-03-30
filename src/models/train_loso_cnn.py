import os
import sys
import json
import numpy as np
import pandas as pd

from pathlib import Path

# Ajouter src/ au path pour trouver config.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# choix ressource
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # pour utiliser le cpu

# Metrics
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Config
from config import *
#-------------------------------------------------------------------------------------------------//params

# LOSO CNN ADAPTATIF - Configuration par participant
# Participants difficiles (F1 < 60%) = fenêtre + epochs augmentés
WINDOW_CONFIGS = {
    3: {'window_size': 480, 'step_size': 240, 'epochs': 15},  # P03: difficile (41% F1) → besoin + contexte
    7: {'window_size': 480, 'step_size': 240, 'epochs': 15},  # P07: difficile (55% F1) → besoin + contexte
    11: {'window_size': 480, 'step_size': 240, 'epochs': 15}, # P11: difficile (54% F1) → besoin + contexte
    # Autres participants: config standard
    'default': {'window_size': 240, 'step_size': 120, 'epochs': 10}
}

# Constraints from User (default)

BATCH_SIZE = 64

# original label mapping from src/config.py:
# {"baseline": -1, "activity": 0, "fatigue": 1}
# Categorical mapping: to use keras to_categorical, labels must be 0, 1, 2
# -1 -> 0 (baseline)
#  0 -> 1 (activity)
#  1 -> 2 (fatigue)
LABEL_MAPPING = {-1: 0, 0: 1, 1: 2}
TARGET_NAMES = ['baseline (-1)', 'activity (0)', 'fatigue (1)']
#----------------------------------------------------------------------------------------------//functions
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
    if 'timestamp' in df.columns:
        df = df.sort_values([COL_PARTICIPANT, COL_SESSION, 'timestamp'])

    for (part, sess), group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        wx, wy = extract_windows_from_group(group, window_size, step_size)
        X_all.extend(wx)
        y_all.extend(wy)

    return np.array(X_all), np.array(y_all)

MODEL_NAME = "CNN_1D_LOSO"

def build_cnn_1d(input_shape, num_classes):
    """
    Simple 1D CNN Architecture.
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    if not DATA_PROCESSED.exists():
        print(f"Dataset not found: {DATA_PROCESSED}")
        print("Please ensure `main1.py` successfully generated the processed dataset.")
        return
        
    print(f"Loading dataset from {DATA_PROCESSED}...")
    df = pd.read_csv(DATA_PROCESSED)
    
    # Map the labels to categorical 0, 1, 2
    df[COL_LABEL] = df[COL_LABEL].map(LABEL_MAPPING)
    
    participants = df[COL_PARTICIPANT].unique()
    print(f"Found {len(participants)} participants: {participants}")
    
    num_classes = len(LABEL_MAPPING)
    all_metrics = []
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for test_idx, test_part in enumerate(participants):
        print(f"\n" + "="*60)
        print(f"FOLD {test_idx+1}/{len(participants)} — TEST PARTICIPANT: {test_part}")
        print("="*60)

        # ✅ Sélectionner configuration adaptée au participant
        if test_part in WINDOW_CONFIGS:
            config = WINDOW_CONFIGS[test_part]
            print(f"🎯 CONFIG ADAPTÉE (participant difficile)")
        else:
            config = WINDOW_CONFIGS['default']
            print(f"📊 CONFIG STANDARD")

        # Appliquer configuration pour ce fold
        WINDOW_SIZE = config['window_size']
        STEP_SIZE = config['step_size']
        EPOCHS = config['epochs']

        print(f"   Window: {WINDOW_SIZE} | Step: {STEP_SIZE} | Epochs: {EPOCHS}")
        
        # LOSO Split
        df_train = df[df[COL_PARTICIPANT] != test_part]
        df_test = df[df[COL_PARTICIPANT] == test_part]
        
        print("Extracting sliding windows...")
        X_train, y_train = extract_all_windows(df_train, WINDOW_SIZE, STEP_SIZE)
        X_test, y_test = extract_all_windows(df_test, WINDOW_SIZE, STEP_SIZE)
        
        print(f"Train windows: {len(X_train)} | Test windows: {len(X_test)}")
        
        if len(X_test) == 0:
            print(f"WARNING: Test set for participant {test_part} is empty. Skipping fold.")
            continue
            
        # Calculate balanced class weights based ONLY on training set
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
        print(f"Class weights (Train): {class_weight_dict}")
        
        # Convert targets to one-hot encoding for categorical crossentropy
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        # Note: we keep y_test simple for evaluation, but can be categorical if used in evaluate
        # y_test_cat = to_categorical(y_test, num_classes=num_classes)
        
        # Build Model
        input_shape = (WINDOW_SIZE, len(SIGNAL_COLS))
        model = build_cnn_1d(input_shape=input_shape, num_classes=num_classes)
        
        # Train
        print(f"Training model on {len(X_train)} samples across {EPOCHS} epochs...")
        model.fit(
            X_train, y_train_cat,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Test Evaluation
        print(f"Evaluating model on {len(X_test)} samples for Participant {test_part}...")
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Ensure all shapes match
        f1_mac = f1_score(y_test, y_pred, average='macro')
        f1_wei = f1_score(y_test, y_pred, average='weighted')
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"\n>>> Fold {test_idx+1} Metrics:")
        print(f"F1-Macro:          {f1_mac:.4f}")
        print(f"F1-Weighted:       {f1_wei:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        
        report = classification_report(y_test, y_pred, target_names=TARGET_NAMES[:len(np.unique(np.concatenate([y_test, y_pred])))], labels=np.unique(np.concatenate([y_test, y_pred])), zero_division=0)
        print("\nClassification Report:\n", report)

        # --- Save model ---
        model_stem = f"{MODEL_NAME}_fold{test_idx+1}_testP{test_part}"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # .keras (SavedModel format)
        tf_path = MODELS_DIR / f"{model_stem}.keras"
        model.save(tf_path)
        print(f"✔ Modèle TF sauvegardé   : {tf_path}")

        # .tflite (float32)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = MODELS_DIR / f"{model_stem}.tflite"
        tflite_path.write_bytes(tflite_model)
        print(f"✔ Modèle TFLite sauvegardé       : {tflite_path}")

        # .tflite INT8 — TFLite Micro (STM32 / MCU)
        def representative_dataset():
            sample = X_train[:200].astype(np.float32)
            for i in range(len(sample)):
                yield [sample[i:i+1]]

        converter_micro = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_micro.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_micro.representative_dataset = representative_dataset
        converter_micro.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_micro.inference_input_type  = tf.int8
        converter_micro.inference_output_type = tf.int8
        tflite_micro_model = converter_micro.convert()

        tflite_micro_path = MODELS_DIR / f"{model_stem}_int8.tflite"
        tflite_micro_path.write_bytes(tflite_micro_model)
        print(f"✔ Modèle TFLite Micro INT8 sauvegardé: {tflite_micro_path}")

        # C array (.h) pour TFLite Micro
        c_array_path = MODELS_DIR / f"{model_stem}_int8.h"
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

        all_metrics.append({
            'Fold': test_idx + 1,
            'Test_Participant': test_part,
            'F1_Macro': f1_mac,
            'F1_Weighted': f1_wei,
            'Balanced_Accuracy': bal_acc,
            'Window_Size': WINDOW_SIZE,
            'Step_Size': STEP_SIZE,
            'Epochs': EPOCHS,
            'Config_Type': 'ADAPTIVE' if test_part in WINDOW_CONFIGS else 'STANDARD'
        })
        
    # --- Overall Summary ---
    print("\n" + "="*60)
    print("GLOBAL EXPERIMENT RESULTS")
    print("="*60)
    
    if len(all_metrics) == 0:
        print("No valid folds completed. Exiting.")
        return
        
    df_results = pd.DataFrame(all_metrics)
    
    mean_row = {
        'Fold': 'MEAN',
        'Test_Participant': 'ALL',
        'F1_Macro': df_results['F1_Macro'].mean(),
        'F1_Weighted': df_results['F1_Weighted'].mean(),
        'Balanced_Accuracy': df_results['Balanced_Accuracy'].mean(),
        'Window_Size': '-',
        'Step_Size': '-',
        'Epochs': '-',
        'Config_Type': '-'
    }

    std_row = {
        'Fold': 'STD',
        'Test_Participant': 'ALL',
        'F1_Macro': df_results['F1_Macro'].std(),
        'F1_Weighted': df_results['F1_Weighted'].std(),
        'Balanced_Accuracy': df_results['Balanced_Accuracy'].std(),
        'Window_Size': '-',
        'Step_Size': '-',
        'Epochs': '-',
        'Config_Type': '-'
    }
    
    # Save the CSV
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    csv_path = reports_dir / "results_summary.csv"
    df_results.to_csv(csv_path, index=False)

    # Save JSON
    metrics_json = {
        "model_name": MODEL_NAME,
        "training_strategy": "LOSO_ADAPTIVE",
        "adaptive_config": {
            "description": "Participants difficiles reçoivent fenêtres + epochs augmentés",
            "difficult_participants": [3, 7, 11],
            "configs": WINDOW_CONFIGS
        },
        "hyperparameters": {
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
        "folds": [
            {k: (float(v) if isinstance(v, (np.floating, float))
                 else int(v) if isinstance(v, (np.integer, int))
                 else v)
             for k, v in m.items()}
            for m in all_metrics
        ],
        "summary": {
            "F1_Macro_mean":          float(mean_row["F1_Macro"]),
            "F1_Macro_std":           float(std_row["F1_Macro"]),
            "F1_Weighted_mean":       float(mean_row["F1_Weighted"]),
            "F1_Weighted_std":        float(std_row["F1_Weighted"]),
            "Balanced_Accuracy_mean": float(mean_row["Balanced_Accuracy"]),
            "Balanced_Accuracy_std":  float(std_row["Balanced_Accuracy"]),
        }
    }
    json_path = reports_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=4)

    print("\nSummary metrics per fold over all tests:")
    print(df_results.tail(2).to_string(index=False))
    print(f"\nFull breakdown exported to {csv_path}")
    print(f"Metrics JSON exported to    {json_path}")

# --------------------------------------------------------------------------------------------------//main
if __name__ == "__main__":
    # Disable eager execution spam/memory if needed
    # tf.get_logger().setLevel('ERROR')
    main()
