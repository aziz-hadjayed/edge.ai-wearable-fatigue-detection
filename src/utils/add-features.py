"""
extract_ppg_ear.py
==================
Extraction features PPG + Génération ambiant synchronisé
- Fréquence PPG: 100 Hz
- Fenêtre: 20s (2000 samples) avec 99% overlap (0.25s)
- Ambiant: 4 Hz synchronisé avec timestamps EDA/PPG
- Output: features_ppg_ear_(left/right).csv + ambient_grandeur.csv
"""

import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

# ═══════════════════════════════════════════════════════════════════════════════
# METEOSTAT (optionnel)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from meteostat import Point, Hourly
    METEOSTAT_AVAILABLE = True
except ImportError:
    METEOSTAT_AVAILABLE = False
    print("WARNING: meteostat non installé. Ambiant avec valeurs par défaut.")
    print("Installez: pip install meteostat")

CAMBRIDGE_LAT = 52.2109
CAMBRIDGE_LON = 0.0917
CAMBRIDGE_ELEVATION = 6


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION PPG
# ═══════════════════════════════════════════════════════════════════════════════
BASE_PATH = "/media/mohamedaziz-hadjayed/D/aziz_data/fatigue_detection/edge-ai-wearable-fatigue-detection/data/01_raw"
EAR_SIDE = "left"

FS = 100.0
WINDOW_SAMPLES = 2000
STEP_SAMPLES = 25

PEAK_MIN_DIST = 15
PEAK_THRESHOLD = 0.3
EPS = 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS AMBIANT (NOUVEAU)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_meteo_two_hours(session_datetime):
    """Récupère température et humidité aux heures H et H+1."""
    if not METEOSTAT_AVAILABLE:
        return None

    location = Point(CAMBRIDGE_LAT, CAMBRIDGE_LON, CAMBRIDGE_ELEVATION)
    hour_h = session_datetime.replace(minute=0, second=0, microsecond=0)
    hour_h1 = hour_h + timedelta(hours=1)

    try:
        data = Hourly(location, hour_h, hour_h1 + timedelta(minutes=1))
        data = data.fetch()
        if len(data) < 2:
            return None
        return {
            'temp0': data['temp'].iloc[0], 'temp1': data['temp'].iloc[1],
            'hum0': data['rhum'].iloc[0], 'hum1': data['rhum'].iloc[1]
        }
    except Exception as e:
        print(f"    [WARN] API Meteostat: {e}")
        return None


def generate_ambient_synced(session_id, session_datetime, timestamp_ref, seed=None):
    """
    Génère ambient_grandeur.csv SYNCHRONISÉ avec les timestamps EDA/PPG.

    Paramètres:
        session_id: ex "P01_S1"
        session_datetime: datetime de début
        timestamp_ref: array des timestamps en ms (provenant de EDA ou PPG)
        seed: reproductibilité

    Retourne:
        DataFrame ambient ou None
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(int(session_datetime.timestamp()) % 10000)

    # Récupérer météo
    meteo = fetch_meteo_two_hours(session_datetime)

    t_seconds = timestamp_ref / 1000.0
    n_points = len(timestamp_ref)

    if meteo:
        # Interpolation linéaire sur timestamps exacts
        temp_base = meteo['temp0'] + (meteo['temp1'] - meteo['temp0']) * (t_seconds / 3600.0)
        hum_base = meteo['hum0'] + (meteo['hum1'] - meteo['hum0']) * (t_seconds / 3600.0)
    else:
        # Valeurs par défaut
        temp_base = np.full(n_points, 45.0)
        hum_base = np.full(n_points, 65.0)

    # BRUIT RÉALISTE
    noise_temp = np.random.normal(0, 0.05, n_points)  # ±0.1°C subtil
    noise_hum  = np.random.normal(0, 0.2, n_points)   # ±0.4% subti

    cycle_clim = 0.5 * np.sin(2 * np.pi * t_seconds / 900)  # ±0.1°C, 15min

    # Perturbations ponctuelles
    n_perturbations = np.random.choice([0, 1])
    perturbation_temp = np.zeros(n_points)

    for _ in range(n_perturbations):
        center_idx = np.random.randint(int(n_points * 0.2), int(n_points * 0.8))
        width = np.random.randint(120, 240)
        amplitude = np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.4)
        x = np.arange(n_points)
        perturbation_temp += amplitude * np.exp(-0.5 * ((x - center_idx) / (width/4))**2)

    # Assemblage
    temperature_amb = temp_base + noise_temp + perturbation_temp #+ cycle_clim
    humidite_amb = hum_base+ noise_hum# - 0.5 * (temperature_amb - temp_base)

    # Contraintes physiques
    temperature_amb = np.clip(temperature_amb, 15.0, 25.0)
    humidite_amb = np.clip(humidite_amb, 30.0, 90.0)

    df_ambient = pd.DataFrame({
        'timestamp': timestamp_ref.astype(int),
        'temperature_amb': np.round(temperature_amb, 3),
        'humidite_amb': np.round(humidite_amb, 2)
    })

    return df_ambient


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS PPG (INCHANGÉES)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_peaks(sig: np.ndarray) -> np.ndarray:
    if len(sig) < 10:
        return np.array([], dtype=int)

    mean_val = np.mean(sig)
    max_val = np.max(sig)
    threshold = mean_val + PEAK_THRESHOLD * (max_val - mean_val)

    is_peak = (
        (sig[2:-2] > sig[1:-3]) & (sig[2:-2] > sig[3:-1]) &
        (sig[2:-2] > sig[0:-4]) & (sig[2:-2] > sig[4:])
    )
    candidates = np.where(is_peak)[0] + 2

    peaks: List[int] = []
    last = -PEAK_MIN_DIST
    for idx in candidates:
        if sig[idx] < threshold:
            continue
        if idx - last < PEAK_MIN_DIST:
            continue
        peaks.append(idx)
        last = idx

    return np.array(peaks, dtype=int)


def calc_rr(peaks: np.ndarray) -> np.ndarray:
    if len(peaks) < 2:
        return np.array([], dtype=float)
    rr = np.diff(peaks) * (1000.0 / FS)
    return rr[(rr >= 300.0) & (rr <= 2000.0)]


def calc_bpm(rr: np.ndarray) -> float:
    if len(rr) < 2:
        return 0.0
    bpm = 60000.0 / np.mean(rr)
    return float(np.clip(bpm, 30.0, 220.0))


def calc_sdnn(rr: np.ndarray) -> float:
    return float(np.std(rr, ddof=0)) if len(rr) >= 3 else 0.0


def calc_rmssd(rr: np.ndarray) -> float:
    if len(rr) < 2:
        return 0.0
    diffs = np.diff(rr)
    return float(np.sqrt(np.mean(diffs ** 2)))


def calc_pnn50(rr: np.ndarray) -> float:
    if len(rr) < 2:
        return 0.0
    diffs = np.abs(np.diff(rr))
    return float(np.sum(diffs > 50.0) / len(diffs) * 100.0)


def calc_lfhf(rr: np.ndarray, smooth_win: int = 3) -> float:
    min_rr = 2 * smooth_win + 1
    if len(rr) < min_rr:
        return 1.0

    half = smooth_win
    rr_smooth = np.zeros_like(rr, dtype=float)
    for i in range(len(rr)):
        start = max(0, i - half)
        end = min(len(rr), i + half + 1)
        rr_smooth[i] = np.mean(rr[start:end])

    rr_hf = rr - rr_smooth
    var_lf = np.var(rr_smooth, ddof=0)
    var_hf = np.var(rr_hf, ddof=0)

    if var_hf < EPS:
        return 2.0

    ratio = var_lf / var_hf
    return float(np.clip(ratio, 0.1, 20.0))


def calc_spo2(ch_a: np.ndarray, ch_b: np.ndarray) -> float:
    if len(ch_a) < 10 or len(ch_b) < 10:
        return 0.0

    dc_a = np.mean(ch_a)
    dc_b = np.mean(ch_b)
    ac_a = np.ptp(ch_a)
    ac_b = np.ptp(ch_b)

    if dc_a < EPS or dc_b < EPS or ac_a < EPS:
        return 0.0

    R = (ac_b / dc_b) / (ac_a / dc_a)
    spo2 = 110.0 - 25.0 * R
    return float(np.clip(spo2, 70.0, 100.0))


def calc_amplitude(sig: np.ndarray) -> float:
    return float(np.ptp(sig)) if len(sig) > 0 else 0.0


def calc_skewness(sig: np.ndarray) -> float:
    if len(sig) < 3:
        return 0.0

    centered = sig - np.mean(sig)
    sum2 = np.sum(centered ** 2)
    sum3 = np.sum(centered ** 3)
    variance = sum2 / len(sig)
    std_dev = np.sqrt(variance)

    if std_dev < EPS:
        return 0.0

    skew = (sum3 / len(sig)) / (std_dev ** 3)
    return float(np.clip(skew, -3.0, 3.0))


def extract_window_features(sig_ir: np.ndarray, sig_red: np.ndarray) -> Dict[str, float]:
    peaks = detect_peaks(sig_ir)
    rr = calc_rr(peaks)

    return {
        "bpm": calc_bpm(rr),
        "sdnn": calc_sdnn(rr),
        "rmssd": calc_rmssd(rr),
        "pnn50": calc_pnn50(rr),
        "lfhf": calc_lfhf(rr),
        "spo2": calc_spo2(sig_ir, sig_red),
        "amplitude": calc_amplitude(sig_ir),
        "skewness": calc_skewness(sig_ir),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER CSV
# ═══════════════════════════════════════════════════════════════════════════════
CSV_HEADER = [
    "timestamp",
    "window_id",
    "bpm",
    "sdnn",
    "rmssd",
    "pnn50",
    "lfhf",
    "spo2",
    "amplitude",
    "skewness",
]


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTION: RÉCUPÉRER DATETIME SESSION (À ADAPTER)
# ═══════════════════════════════════════════════════════════════════════════════

def get_session_datetime(participant_id: str, session_id: str) -> datetime:
    """
    Récupère la date/heure réelle de la session.

    À ADAPTER selon votre structure de données:
    - Option 1: Lire metadata.json
    - Option 2: Extraire du nom de fichier
    - Option 3: Base de données externe

    Par défaut: génère datetime fictive (À REMPLACER !)
    """
    # Option 1: metadata.json
    metadata_path = os.path.join(BASE_PATH, participant_id, session_id, "metadata.json")
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path) as f:
            meta = json.load(f)
            if 'datetime' in meta:
                return datetime.fromisoformat(meta['datetime'])

    # Option 2: fichier timestamp_ref.csv
    ts_path = os.path.join(BASE_PATH, participant_id, session_id, "session_info.csv")
    if os.path.exists(ts_path):
        df_info = pd.read_csv(ts_path)
        if 'datetime' in df_info.columns:
            return datetime.fromisoformat(df_info['datetime'].iloc[0])

    # FALLBACK (À REMPLACER PAR VOTRE LOGIQUE RÉELLE)
    day = int(participant_id)
    hour = 9 + (int(session_id) - 1) * 3
    return datetime(2024, 3, 15 + day - 1, hour, 0, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL - 36 SESSIONS + AMBIANT SYNCHRONISÉ
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("EXTRACTION PPG + GÉNÉRATION AMBIANT SYNCHRONISÉ")
    print("=" * 70)
    print(f"Oreille: {EAR_SIDE.upper()} | Fréquence PPG: {FS} Hz")
    print(f"Fenêtre: {WINDOW_SAMPLES/FS}s | Step: {STEP_SAMPLES/FS}s")
    print("-" * 70)

    total_sessions = 0
    total_windows = 0
    total_ambient = 0

    # BOUCLE 36 SESSIONS
    for participant in range(1, 13):
        participant_id = f"{participant:02d}"
        participant_path = os.path.join(BASE_PATH, participant_id)

        if not os.path.exists(participant_path):
            continue

        for session in range(1, 4):
            session_num = f"{session:02d}"
            session_path = os.path.join(participant_path, session_num)
            session_full_id = f"P{participant_id}_S{session_num}"

            if not os.path.exists(session_path):
                continue

            print(f"\n[{session_full_id}] Traitement...")

            # ─── LECTURE PPG ─────────────────────────────────
            ppg_file = os.path.join(session_path, f"ear_ppg_{EAR_SIDE}.csv")

            if not os.path.exists(ppg_file):
                print(f"  [SKIP] PPG manquant: {ppg_file}")
                continue

            try:
                df_ppg = pd.read_csv(ppg_file)
            except Exception as e:
                print(f"  [ERR] Lecture PPG: {e}")
                continue

            required_cols = ["timestamp", "green", "ir", "red"]
            missing = [c for c in required_cols if c not in df_ppg.columns]
            if missing:
                print(f"  [SKIP] Colonnes manquantes: {missing}")
                continue

            timestamps = df_ppg["timestamp"].values
            sig_ir = df_ppg["ir"].values
            sig_red = df_ppg["red"].values

            print(f"  → PPG: {len(timestamps)} points | ts={timestamps[0]}→{timestamps[-1]}ms")

            # ─── GÉNÉRATION AMBIANT (SYNCHRONISÉ) ────────────
            session_dt = get_session_datetime(participant_id, session_num)
            print(f"  → Session datetime: {session_dt.strftime('%Y-%m-%d %H:%M')}")

            df_ambient = generate_ambient_synced(
                session_id=session_full_id,
                session_datetime=session_dt,
                timestamp_ref=timestamps  # ← SYNCHRONISÉ AVEC PPG
            )

            if df_ambient is not None:
                # Sauvegarde ambiant
                ambient_dir = os.path.join(session_path)
                os.makedirs(ambient_dir, exist_ok=True)
                ambient_path = os.path.join(ambient_dir, "ambient_grandeur.csv")
                df_ambient.to_csv(ambient_path, index=False)

                print(f"  ✓ Ambiant: {ambient_path}")
                print(f"    Temp: {df_ambient['temperature_amb'].mean():.2f}°C (±{df_ambient['temperature_amb'].std():.2f})")
                print(f"    Hum:  {df_ambient['humidite_amb'].mean():.1f}% (±{df_ambient['humidite_amb'].std():.1f})")
                total_ambient += 1

            # ─── FENÊTRAGE PPG ───────────────────────────────
            n_samples = min(len(sig_ir), len(sig_red), len(timestamps))
            n_windows = (n_samples - WINDOW_SAMPLES) // STEP_SAMPLES + 1

            if n_windows <= 0:
                print(f"  [SKIP] Pas assez de données")
                continue

            # Écriture features PPG
            output_file = os.path.join(session_path, f"features_ppg_ear_{EAR_SIDE}.csv")

            with open(output_file, "w", newline="") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=CSV_HEADER)
                writer.writeheader()

                for w in range(n_windows):
                    start = w * STEP_SAMPLES
                    end = start + WINDOW_SAMPLES

                    window_timestamp = timestamps[start]

                    feats = extract_window_features(
                        sig_ir[start:end],
                        sig_red[start:end]
                    )

                    row = {
                        "timestamp": window_timestamp,
                        "window_id": w,
                        **feats
                    }
                    writer.writerow(row)

            total_sessions += 1
            total_windows += n_windows
            print(f"  ✓ PPG features: {n_windows} fenêtres → {output_file}")

    print("\n" + "=" * 70)
    print("RÉSULTATS")
    print("=" * 70)
    print(f"Sessions PPG traitées: {total_sessions}")
    print(f"Fenêtres PPG extraites: {total_windows}")
    print(f"Fichiers ambiant générés: {total_ambient}")
    print("=" * 70)


if __name__ == "__main__":
    main()
