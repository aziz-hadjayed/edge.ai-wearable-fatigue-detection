# ------------------------------------------// Importations

from src.config import *

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks






OUTPUT_DIR = Path("../data/02_interim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
#    "chest_physiology_summary.csv": {"hr": "chest_hr","br": "chest_br","posture": "chest_posture","hrv": "chest_hrv",},
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
        return []

    # 🔥 Convertir en temps relatif
    t0 = intervals[0][0]
    intervals = [(s - t0, e - t0, l) for s, e, l in intervals]

    print("  ℹ Intervalles (relatif):", intervals)
    return intervals


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Resample → garder temps relatif
# ═══════════════════════════════════════════════════════════════════════════════
def resample_to_4hz(df, native_freq):
    df = df.copy()

    # 🔥 convertir en temps relatif
    df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]

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

    # 🔥 retour en ms relatif
    df["timestamp"] = (df.index - df.index[0]).total_seconds() * 1000

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Load + clean
# ═══════════════════════════════════════════════════════════════════════════════
def load_rename_resample(filepath, col_map, freq):
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()

        ts_col = [c for c in df.columns if "time" in c][0]
        df = df.rename(columns={ts_col: "timestamp"})

        cols = {"timestamp": "timestamp"}
        for k, v in col_map.items():
            if k in df.columns:
                cols[k] = v

        df = df[list(cols.keys())].rename(columns=cols)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna().sort_values("timestamp")

        return resample_to_4hz(df, freq)

    except Exception as e:
        print("⚠ erreur:", filepath.name, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Breathing RPM
# ═══════════════════════════════════════════════════════════════════════════════
def compute_breathing_rpm(path):
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

    return resample_to_4hz(df, 0.5)


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
