import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Désactive le GPU pour éviter les erreurs de drivers
import numpy as np
import pandas as pd
from pathlib import Path

# Metrics
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import sys



# Config
from config import *

# Constraints from User
WINDOW_SIZE = 240
STEP_SIZE = 120
EPOCHS = 10
BATCH_SIZE = 64

# original label mapping from src/config.py:
# {"baseline": -1, "activity": 0, "fatigue": 1}
# Categorical mapping: to use keras to_categorical, labels must be 0, 1, 2
# -1 -> 0 (baseline)
#  0 -> 1 (activity)
#  1 -> 2 (fatigue)
LABEL_MAPPING = {-1: 0, 0: 1, 1: 2}
TARGET_NAMES = ['baseline (-1)', 'activity (0)', 'fatigue (1)']

def extract_windows_from_group(group_df):
    """
    Extract overlapping continuous windows from a single session.
    No timestamp used in features. Label is the majority vote.
    """
    X_group = group_df[SIGNAL_COLS].values
    y_group = group_df[COL_LABEL].values
    
    windows_x = []
    windows_y = []
    
    n_samples = len(X_group)
    for start in range(0, n_samples - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window_x = X_group[start:end]
        window_y_raw = y_group[start:end]
        
        # Majority vote for label
        vals, counts = np.unique(window_y_raw, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_label = vals[majority_idx]
        
        windows_x.append(window_x)
        windows_y.append(majority_label)
        
    return windows_x, windows_y

def extract_all_windows(df):
    """
    Group by participant and session to ensure NO windows overlap across sessions.
    """
    X_all = []
    y_all = []
    
    # Sort optionally by participant, session, timestamp to be safe
    if 'timestamp' in df.columns:
        df = df.sort_values([COL_PARTICIPANT, COL_SESSION, 'timestamp'])
        
    for (part, sess), group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        wx, wy = extract_windows_from_group(group)
        X_all.extend(wx)
        y_all.extend(wy)
        
    return np.array(X_all), np.array(y_all)

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
        
        # LOSO Split
        df_train = df[df[COL_PARTICIPANT] != test_part]
        df_test = df[df[COL_PARTICIPANT] == test_part]
        
        print("Extracting sliding windows...")
        X_train, y_train = extract_all_windows(df_train)
        X_test, y_test = extract_all_windows(df_test)
        
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
        
        all_metrics.append({
            'Fold': test_idx + 1,
            'Test_Participant': test_part,
            'F1_Macro': f1_mac,
            'F1_Weighted': f1_wei,
            'Balanced_Accuracy': bal_acc
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
        'Balanced_Accuracy': df_results['Balanced_Accuracy'].mean()
    }
    
    std_row = {
        'Fold': 'STD',
        'Test_Participant': 'ALL',
        'F1_Macro': df_results['F1_Macro'].std(),
        'F1_Weighted': df_results['F1_Weighted'].std(),
        'Balanced_Accuracy': df_results['Balanced_Accuracy'].std()
    }
    
    # Save the CSV
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    csv_path = reports_dir / "results_summary.csv"
    df_results.to_csv(csv_path, index=False)
    
    print("\nSummary metrics per fold over all tests:")
    print(df_results.tail(2).to_string(index=False))
    print(f"\nFull breakdown exported to {csv_path}")


if __name__ == "__main__":
    # Disable eager execution spam/memory if needed
    # tf.get_logger().setLevel('ERROR')
    main()
