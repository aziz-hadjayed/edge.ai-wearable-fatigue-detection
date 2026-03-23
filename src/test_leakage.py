import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Désactive le GPU pour éviter les erreurs de drivers
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Config locale
from src.config import DATA_PROCESSED, SIGNAL_COLS, COL_PARTICIPANT, COL_SESSION, COL_LABEL

WINDOW_SIZE = 240
STEP_SIZE = 240 # On réduit le nombre de fenêtres pour que le test soit rapide
LABEL_MAPPING = {-1: 0, 0: 1, 1: 2}

def get_model():
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(WINDOW_SIZE, len(SIGNAL_COLS))),
        MaxPooling1D(2),
        Flatten(),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def extract_windows(df):
    X, y = [], []
    for (p, s), group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        sig = group[SIGNAL_COLS].values
        lbl = group[COL_LABEL].values
        for i in range(0, len(sig) - WINDOW_SIZE + 1, STEP_SIZE):
            X.append(sig[i:i+WINDOW_SIZE])
            vals, counts = np.unique(lbl[i:i+WINDOW_SIZE], return_counts=True)
            y.append(vals[np.argmax(counts)])
    return np.array(X), np.array(y)

def run_loso(df, leak_mode=False):
    participants = df[COL_PARTICIPANT].unique()
    scores = []
    
    # Pré-normalisation globale (LEAKAGE)
    if leak_mode:
        scaler = RobustScaler()
        df[SIGNAL_COLS] = scaler.fit_transform(df[SIGNAL_COLS])
        print("!!! MODE LEAKAGE ACTIVE : Normalisation globale effectuée !!!")

    for p_test in participants:
        df_train = df[df[COL_PARTICIPANT] != p_test].copy()
        df_test = df[df[COL_PARTICIPANT] == p_test].copy()
        
        # Normalisation locale (CLEAN)
        if not leak_mode:
            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])
            
        X_train, y_train = extract_windows(df_train)
        X_test, y_test = extract_windows(df_test)
        
        y_train_cat = to_categorical(y_train, 3)
        
        model = get_model()
        model.fit(X_train, y_train_cat, epochs=3, batch_size=64, verbose=0)
        
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        scores.append(balanced_accuracy_score(y_test, y_pred))
        print(f"Fold {p_test} done.")
        
    return np.mean(scores)

if __name__ == "__main__":
    raw_df = pd.read_csv(DATA_PROCESSED)
    raw_df[COL_LABEL] = raw_df[COL_LABEL].map(LABEL_MAPPING)

    print("\n--- TEST 1 : APPROCHE CLEAN (Sans fuite) ---")
    mean_clean = run_loso(raw_df.copy(), leak_mode=False)
    
    print("\n--- TEST 2 : APPROCHE LEAKAGE (Avec fuite) ---")
    mean_leak = run_loso(raw_df.copy(), leak_mode=True)
    
    print("\n" + "="*40)
    print(f"RÉSULTAT DU TEST DE FUITE :")
    print(f"Précision approche CLEAN   : {mean_clean:.4f}")
    print(f"Précision approche LEAKAGE : {mean_leak:.4f}")
    print(f"Différence (Biais)         : {mean_leak - mean_clean:.4f}")
    print("="*40)
    if mean_leak > mean_clean:
        print("CONSTAT : Le score avec leakage est supérieur. Ton modèle 'triche' en utilisant des stats globales.")
