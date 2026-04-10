# --------------------------------------------------------------------// Importations

from src.config import *

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks



# Ces variables ne sont plus utilisées directement pour sauvegarde ici — gérées dans main.py
# mais gardées pour cohérence si besoin de debug local
# OUTPUT_DIR = INTERIM_DIR
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PATH_BREATHING = "chest_raw_breathing.csv"
PATH_MARKERS = "exp_markers.csv"

# ── Paramètres respiration ────────────────────────────────────────────────────
FS_BREATHING = 25.0
FMIN, FMAX   = 0.05, 0.80
MIN_DIST_S   = 1.5
PROM_MIN     = 100
PROM_OK      = 500

# ── Fréquences capteurs ───────────────────────────────────────────────────────
SENSOR_FREQ = {
    "chest_physiology_summary.csv": 1,
    "wrist_acc.csv":                32,
    "wrist_eda.csv":                4,
    "wrist_hr.csv":                 1,
    "wrist_ibi.csv":                0.59,
    "wrist_skin_temperature.csv":   4,
}

TARGET_FREQ   = 4
TARGET_PERIOD = 250  # ms

# ── Colonnes ──────────────────────────────────────────────────────────────────
FILE_COLS = {

    "wrist_acc.csv": {"ax": "acc_x", "ay": "acc_y", "az": "acc_z"},
    "wrist_eda.csv": {"eda": "eda"},
    "wrist_hr.csv": {"hr": "wrist_hr"},
    "wrist_ibi.csv": {"duration": "ibi"},
    "wrist_skin_temperature.csv": {"temp": "temp"},
}

LABELS_OF_INTEREST = ["baseline", "activity", "fatigue"]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Labels → convertir en temps relatif
# ═══════════════════════════════════════════════════════════════════════════════
def get_label_intervals(markers_path):
    df = pd.read_csv(markers_path)
    df.columns = df.columns.str.strip()

    intervals = []
    active = {}

    for _, row in df.iterrows():
        ts = float(row["utcTime"])
        event = str(row["eventMarker"]).lower()

        for lbl in LABELS_OF_INTEREST:
            if f"start_{lbl}" in event:
                active[lbl] = ts
            elif f"end_{lbl}" in event and lbl in active:
                intervals.append((active.pop(lbl), ts, lbl))

    if not intervals:
        return [], None

    # 🔥 Référence de début de session (absolue)
    t0 = intervals[0][0]
    # 🔥 Convertir en temps relatif par rapport à t0
    intervals_rel = [(s - t0, e - t0, l) for s, e, l in intervals]

    print(f"  ℹ t0 référence: {t0}")
    print("  ℹ Intervalles (relatif):", intervals_rel)
    return intervals_rel, t0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Resample → garder temps relatif
# ═══════════════════════════════════════════════════════════════════════════════
def resample_to_4hz(df, native_freq, t_ref):
    df = df.copy()

    # 🔥 Synchronisation : rendre les timestamps relatifs à t_ref (absolu)
    # AVANT resampling pour que tous les capteurs soient alignés sur le même référentiel
    df["timestamp"] = df["timestamp"] - t_ref

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("datetime").drop(columns=["timestamp"])

    period = "250ms"

    if native_freq > TARGET_FREQ:
        df = df.resample(period).mean()
    elif native_freq < 1:
        df = df.resample(period).ffill()
    elif native_freq < TARGET_FREQ:
        df = df.resample(period).interpolate()
    else:
        df = df.resample(period).mean()

    # 🔥 retour en ms relatif à t_ref (1970-01-01 00:00:00 dans datetime est notre 0 relatif)
    df["timestamp"] = (df.index - pd.to_datetime(0, unit="ms")).total_seconds() * 1000

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Load + clean
# ═══════════════════════════════════════════════════════════════════════════════
def load_rename_resample(filenames, col_map, freq, t_ref):
    """
    filenames : Liste de un ou plusieurs fichiers (shards) pour un même capteur
    """
    try:
        all_dfs = []
        for filepath in filenames:
            df_part = pd.read_csv(filepath)
            df_part.columns = df_part.columns.str.strip().str.lower()
            all_dfs.append(df_part)
        
        df = pd.concat(all_dfs)
        ts_col = [c for c in df.columns if "time" in c][0]
        df = df.rename(columns={ts_col: "timestamp"})

        cols = {"timestamp": "timestamp"}
        for k, v in col_map.items():
            if k in df.columns:
                cols[k] = v

        df = df[list(cols.keys())].rename(columns=cols)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna().sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        return resample_to_4hz(df, freq, t_ref)

    except Exception as e:
        print(f"⚠ erreur sur les fichiers {filenames}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Breathing RPM
# ═══════════════════════════════════════════════════════════════════════════════
def compute_breathing_rpm(path, t_ref):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    ts = df["timestamp"].values
    sig = df["breathing_waveform"].values

    sig = sig - np.mean(sig)
    b, a = butter(2, [FMIN, FMAX], btype='band', fs=FS_BREATHING)
    sig = filtfilt(b, a, sig)

    peaks, props = find_peaks(
        sig,
        distance=int(MIN_DIST_S * FS_BREATHING),
        prominence=PROM_MIN
    )

    rows = []
    for i in range(len(peaks)-1):
        dt = (peaks[i+1] - peaks[i]) / FS_BREATHING
        rpm = 60 / dt

        rows.append({
            "timestamp": ts[peaks[i]],
            "breathing_rpm": rpm,
            "breathing_q": int(props["prominences"][i] >= PROM_OK)
        })

    df = pd.DataFrame(rows)

    return resample_to_4hz(df, 0.5, t_ref)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Labels
# ═══════════════════════════════════════════════════════════════════════════════
def assign_labels(df, intervals):
    df["label"] = None

    for s, e, l in intervals:
        mask = (df["timestamp"] >= s) & (df["timestamp"] <= e)
        df.loc[mask, "label"] = l

    print("  ℹ labels trouvés:", df["label"].value_counts(dropna=False))

    return df.dropna(subset=["label"])
