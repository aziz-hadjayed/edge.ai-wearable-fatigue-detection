from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

# Add project root to sys.path so we can import from src
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import RAW_DIR, PROCESSED_DIR

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None


V1_CANDIDATES = [
    PROCESSED_DIR / "dataset_ref.csv",
]
SIGNALS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "eda",
    "wrist_hr",
    "ibi",
    "temp",
    "breathing_rpm",
    "temperature_amb",
    "humidite_amb",
    "age",
    "gender",
    "ear_ppg_left_green",
    "ear_ppg_left_ir",
    "ear_ppg_left_red",
    "ear_ppg_right_green",
    "ear_ppg_right_ir",
    "ear_ppg_right_red",
    "bpm_left",
    "sdnn_left",
    "rmssd_left",
    "pnn50_left",
    "lfhf_left",
    "spo2_left",
    "amplitude_left",
    "skewness_left",
    "bpm_right",
    "sdnn_right",
    "rmssd_right",
    "pnn50_right",
    "lfhf_right",
    "spo2_right",
    "amplitude_right",
    "skewness_right",
]

SIGNAL_UNITS = {
    "acc_x": "g",
    "acc_y": "g",
    "acc_z": "g",
    "eda": "uS",
    "wrist_hr": "bpm",
    "ibi": "ms",
    "temp": "deg C",
    "breathing_rpm": "rpm",
    "temperature_amb":"deg C",
    "humidite_amb":"%",
    "age": "ans",
    "gender": "code",
    "ear_ppg_left_green":"signal",
    "ear_ppg_left_ir":"signal",
    "ear_ppg_left_red":"signal",
    "ear_ppg_right_green":"signal",
    "ear_ppg_right_ir":"signal",
    "ear_ppg_right_red":"signal",
    "bpm_left":"bpm",
    "sdnn_left":"ms",
    "rmssd_left":"ms",
    "pnn50_left":"%",
    "lfhf_left":"-",
    "spo2_left":"%",
    "amplitude_left":"Sortie du ADC (convertisseur)",
    "skewness_left":"-",
    "bpm_right":"bpm",
    "sdnn_right":"ms",
    "rmssd_right":"ms",
    "pnn50_right":"%",
    "lfhf_right":"-",
    "spo2_right":"%",
    "amplitude_right":"Sortie du ADC (convertisseur)",
    "skewness_right":"-",
}

FILE_COLS = {
    "wrist_acc.csv": {"ax": "acc_x", "ay": "acc_y", "az": "acc_z"},
    "wrist_eda.csv": {"eda": "eda"},
    "wrist_hr.csv": {"hr": "wrist_hr"},
    "wrist_ibi.csv": {"duration": "ibi"},
    "wrist_skin_temperature.csv": {"temp": "temp"},
    "ambient_grandeur.csv": {"temperature_amb": "temperature_amb", "humidite_amb": "humidite_amb"},
    "ear_ppg_left.csv":{"green":"ear_ppg_left_green","ir":"ear_ppg_left_ir","red":"ear_ppg_left_red"},
    "ear_ppg_right.csv":{"green":"ear_ppg_right_green","ir":"ear_ppg_right_ir","red":"ear_ppg_right_red"},
    "features_ppg_ear_left.csv":{"amplitude":"amplitude_left","skewness":"skewness_left","bpm":"bpm_left","sdnn":"sdnn_left","rmssd":"rmssd_left","pnn50":"pnn50_left","lfhf":"lfhf_left","spo2":"spo2_left"},
    "features_ppg_ear_right.csv":{"amplitude":"amplitude_right","skewness":"skewness_right","bpm":"bpm_right","sdnn":"sdnn_right","rmssd":"rmssd_right","pnn50":"pnn50_right","lfhf":"lfhf_right","spo2":"spo2_right"},
}

SIGNAL_FILES = {
    "acc_x": "wrist_acc.csv",
    "acc_y": "wrist_acc.csv",
    "acc_z": "wrist_acc.csv",
    "eda": "wrist_eda.csv",
    "wrist_hr": "wrist_hr.csv",
    "ibi": "wrist_ibi.csv",
    "temp": "wrist_skin_temperature.csv",
    "breathing_rpm": "chest_raw_breathing.csv",
    "temperature_amb": "ambient_grandeur.csv",
    "humidite_amb": "ambient_grandeur.csv",
    "ear_ppg_left_green": "ear_ppg_left.csv",
    "ear_ppg_left_ir": "ear_ppg_left.csv",
    "ear_ppg_left_red": "ear_ppg_left.csv",
    "ear_ppg_right_green": "ear_ppg_right.csv",
    "ear_ppg_right_ir": "ear_ppg_right.csv",
    "ear_ppg_right_red": "ear_ppg_right.csv",
    "amplitude_left": "features_ppg_ear_left.csv",
    "skewness_left": "features_ppg_ear_left.csv",
    "bpm_left": "features_ppg_ear_left.csv",
    "sdnn_left": "features_ppg_ear_left.csv",
    "rmssd_left": "features_ppg_ear_left.csv",
    "pnn50_left": "features_ppg_ear_left.csv",
    "lfhf_left": "features_ppg_ear_left.csv",
    "spo2_left": "features_ppg_ear_left.csv",
    "amplitude_right": "features_ppg_ear_right.csv",
    "skewness_right": "features_ppg_ear_right.csv",
    "bpm_right": "features_ppg_ear_right.csv",
    "sdnn_right": "features_ppg_ear_right.csv",
    "rmssd_right": "features_ppg_ear_right.csv",
    "pnn50_right": "features_ppg_ear_right.csv",
    "lfhf_right": "features_ppg_ear_right.csv",
    "spo2_right": "features_ppg_ear_right.csv",

}

LABELS = {
    -1: {"name": "baseline", "color": "rgba(65, 105, 225, 0.14)"},
    0: {"name": "activity", "color": "rgba(46, 160, 67, 0.13)"},
    1: {"name": "fatigue", "color": "rgba(220, 53, 69, 0.14)"},
}
LABEL_MAP = {"baseline": -1, "activity": 0, "fatigue": 1}
GENDER_MAP = {"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0}


def signal_label(signal: str, sampling_rates: dict[str, float] | None = None) -> str:
    """Construit un libelle avec unite et frequence d'echantillonnage."""
    unit = SIGNAL_UNITS.get(signal, "")
    label = f"{signal} ({unit})" if unit else signal
    if sampling_rates and signal in sampling_rates and pd.notna(sampling_rates[signal]) and sampling_rates[signal] > 0:
        label = f"{label} - {sampling_rates[signal]:.2f} Hz"
    return label


def find_v1_dataset() -> Path:
    """Trouve le CSV consolidé le plus probable pour la Version 1."""
    for path in V1_CANDIDATES:
        if path.exists():
            return path
    return V1_CANDIDATES[0]


def to_relative_seconds(values: pd.Series, start_ms: float | None = None) -> pd.Series:
    """Convertit des timestamps UTC en millisecondes vers des secondes relatives."""
    numeric = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    if cast(pd.Series, numeric.dropna()).empty:
        return numeric
    if start_ms is None:
        start_ms = float(numeric.min())
    if float(numeric.max()) > 1_000_000_000:
        return cast(pd.Series, (numeric - start_ms) / 1000.0)
    return cast(pd.Series, numeric - float(numeric.min()))


def relative_time_to_seconds(values: pd.Series) -> pd.Series:
    """Convertit un temps relatif en secondes quand il est stocke en millisecondes."""
    numeric = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    diffs = cast(pd.Series, numeric.sort_values().diff().dropna())
    diffs = diffs[diffs > 0]
    median_diff = float(diffs.median()) if not diffs.empty else np.nan
    relative = cast(pd.Series, numeric - float(numeric.min()))
    if pd.notna(median_diff) and median_diff > 10:
        return cast(pd.Series, relative / 1000.0)
    if float(numeric.max(skipna=True)) > 10_000:
        return cast(pd.Series, relative / 1000.0)
    return relative


def estimate_dataframe_sampling_rates(df: pd.DataFrame, signals: list[str]) -> dict[str, float]:
    """Estime les frequences apres consolidation quand les donnees sont deja fusionnees."""
    if "timestamp" not in df.columns or df.empty:
        return {}
    timestamps = relative_time_to_seconds(df["timestamp"])
    diffs = cast(pd.Series, timestamps.sort_values().diff().dropna())
    diffs = diffs[diffs > 0]
    base_rate = float(1.0 / diffs.median()) if not diffs.empty else np.nan
    return {signal: base_rate for signal in signals if signal in df.columns}


def load_v1(participant: str, session: str) -> tuple[pd.DataFrame, list[str]]:
    """Charge le fichier CSV unique et filtre un participant/session."""
    warnings: list[str] = []
    path = find_v1_dataset()
    if not path.exists():
        return pd.DataFrame(), [f"Fichier V1 introuvable: {path}"]

    df = pd.read_csv(path)
    assert isinstance(df, pd.DataFrame)
    missing = [col for col in ["participant", "session", "timestamp"] if col not in df.columns]
    if missing:
        return pd.DataFrame(), [f"Colonnes V1 manquantes: {', '.join(missing)}"]

    df["participant"] = df["participant"].astype(str).str.zfill(2)
    df["session"] = df["session"].astype(str).str.zfill(2)
    df = df.loc[(df["participant"] == participant) & (df["session"] == session)].copy()
    assert isinstance(df, pd.DataFrame)
    if df.empty:
        return df, [f"Aucune ligne V1 pour participant {participant}, session {session}."]

    df["timestamp"] = relative_time_to_seconds(df["timestamp"])
    for signal in SIGNALS + ["label"]:
        if signal not in df.columns:
            df[signal] = np.nan
            warnings.append(f"Colonne absente en V1, remplie par NaN: {signal}")
    df = df.sort_values("timestamp")
    assert isinstance(df, pd.DataFrame)
    df.attrs["sampling_rates"] = estimate_dataframe_sampling_rates(df, SIGNALS)
    return df, warnings


def read_sensor_file(path: Path, rename_map: dict[str, str], start_ms: float) -> tuple[pd.DataFrame, str | None]:
    """Lit un fichier capteur, renomme ses colonnes et convertit le temps."""
    if not path.exists():
        return pd.DataFrame(), f"Fichier manquant: {path.name}"

    df = pd.read_csv(path)
    assert isinstance(df, pd.DataFrame)
    if "timestamp" not in df.columns:
        return pd.DataFrame(), f"Colonne timestamp absente: {path.name}"

    available = {src: dst for src, dst in rename_map.items() if src in df.columns}
    if not available:
        return pd.DataFrame(), f"Aucune colonne utile dans {path.name}"

    df = df.loc[:, ["timestamp", *available.keys()]].rename(columns=available)
    assert isinstance(df, pd.DataFrame)
    df["timestamp"] = to_relative_seconds(df["timestamp"], start_ms)
    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    res_df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    assert isinstance(res_df, pd.DataFrame)
    return res_df, None


def estimate_sampling_rate(path: Path, start_ms: float) -> float:
    """Estime la frequence d'un capteur a partir des timestamps bruts."""
    if not path.exists():
        return np.nan
    try:
        timestamps = pd.read_csv(path, usecols=["timestamp"])["timestamp"]
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError):
        return np.nan

    relative_time = to_relative_seconds(timestamps, start_ms).dropna().sort_values()
    diffs = cast(pd.Series, relative_time.diff().dropna())
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return np.nan
    return float(1.0 / diffs.median())


def session_sampling_rates(session_dir: Path, start_ms: float) -> dict[str, float]:
    """Calcule la frequence d'origine de chaque signal de la session V2."""
    rates: dict[str, float] = {}
    cache: dict[str, float] = {}
    for signal, filename in SIGNAL_FILES.items():
        if filename not in cache:
            cache[filename] = estimate_sampling_rate(session_dir / filename, start_ms)
        rates[signal] = cache[filename]
    rates["age"] = 0.0
    rates["gender"] = 0.0
    return rates


def load_metadata(participant: str) -> tuple[float, float]:
    """Récupère age et gender depuis metadata.csv si le fichier est présent."""
    metadata_path = RAW_DIR / "metadata.csv"
    if not metadata_path.exists():
        return np.nan, np.nan

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    if "participant_id" not in metadata.columns:
        return np.nan, np.nan

    row = metadata[metadata["participant_id"].astype(str).str.zfill(2) == participant]
    if row.empty:
        return np.nan, np.nan

    age = pd.to_numeric(row.iloc[0].get("age", np.nan), errors="coerce")
    gender_value = str(row.iloc[0].get("gender", "")).strip().lower()
    gender = GENDER_MAP.get(gender_value, np.nan)
    return age if pd.notna(age) else np.nan, gender


def session_start_ms(session_dir: Path) -> float | None:
    """Détermine le début de session à partir des markers puis des capteurs."""
    starts: list[float] = []
    markers_path = session_dir / "exp_markers.csv"
    if markers_path.exists():
        markers = pd.read_csv(markers_path)
        if "utcTime" in markers.columns:
            starts.extend([float(x) for x in cast(pd.Series, pd.to_numeric(markers["utcTime"], errors="coerce")).dropna()])
    for filename in [*FILE_COLS.keys(), "chest_raw_breathing.csv"]:
        path = session_dir / filename
        if path.exists():
            sample = pd.read_csv(path, usecols=["timestamp"])
            starts.extend([float(x) for x in cast(pd.Series, pd.to_numeric(sample["timestamp"], errors="coerce")).dropna().head(1)])
    return min(starts) if starts else None


def compute_breathing_rpm(session_dir: Path, base_time: pd.Series, start_ms: float) -> tuple[pd.DataFrame, str | None]:
    """Calcule breathing_rpm depuis chest_raw_breathing.csv quand c'est possible."""
    path = session_dir / "chest_raw_breathing.csv"
    if not path.exists():
        return pd.DataFrame({"timestamp": base_time, "breathing_rpm": np.nan}), "Fichier manquant: chest_raw_breathing.csv"

    df = pd.read_csv(path)
    assert isinstance(df, pd.DataFrame)
    if "timestamp" not in df.columns:
        return pd.DataFrame({"timestamp": base_time, "breathing_rpm": np.nan}), "Colonne timestamp absente: chest_raw_breathing.csv"

    df["timestamp"] = to_relative_seconds(df["timestamp"], start_ms)
    if "breathing_rpm" in df.columns:
        out = df.loc[:, ["timestamp", "breathing_rpm"]].copy()
        assert isinstance(out, pd.DataFrame)
        out["breathing_rpm"] = pd.to_numeric(out["breathing_rpm"], errors="coerce")
        res_out = out.sort_values("timestamp")
        assert isinstance(res_out, pd.DataFrame)
        return res_out, None

    value_cols = [col for col in df.columns if col != "timestamp"]
    if not value_cols or find_peaks is None:
        return pd.DataFrame({"timestamp": base_time, "breathing_rpm": np.nan}), "Impossible de calculer breathing_rpm."

    waveform = cast(pd.Series, pd.to_numeric(df[value_cols[0]], errors="coerce")).interpolate(limit_direction="both")
    if cast(pd.Series, waveform.dropna()).shape[0] < 3:
        return pd.DataFrame({"timestamp": base_time, "breathing_rpm": np.nan}), "Signal breathing insuffisant."

    dt = df["timestamp"].diff().median()
    sampling_rate = 1.0 / dt if pd.notna(dt) and dt > 0 else 25.0
    distance = max(1, int(sampling_rate * 1.5))
    peaks, _ = find_peaks(waveform.to_numpy(), distance=distance)
    if len(peaks) < 2:
        return pd.DataFrame({"timestamp": base_time, "breathing_rpm": np.nan}), "Pics breathing insuffisants."

    peak_times = df["timestamp"].iloc[peaks].to_numpy()
    intervals = np.diff(peak_times)
    rpm = 60.0 / intervals
    rpm_times = peak_times[1:]
    interpolated = np.interp(base_time, rpm_times, rpm, left=np.nan, right=np.nan)
    return pd.DataFrame({"timestamp": base_time, "breathing_rpm": interpolated}), None


def label_intervals(session_dir: Path, start_ms: float) -> list[tuple[float, float, int]]:
    """Parse exp_markers.csv et retourne les intervalles labelises."""
    markers_path = session_dir / "exp_markers.csv"
    if not markers_path.exists():
        return []

    markers = pd.read_csv(markers_path)
    if not {"utcTime", "eventMarker"}.issubset(markers.columns):
        return []

    markers["timestamp"] = to_relative_seconds(markers["utcTime"], start_ms)
    intervals: list[tuple[float, float, int]] = []
    for phase, label in LABEL_MAP.items():
        starts = markers[markers["eventMarker"].eq(f"start_{phase}")]["timestamp"].tolist()
        ends = markers[markers["eventMarker"].eq(f"end_{phase}")]["timestamp"].tolist()
        for start, end in zip(starts, ends):
            intervals.append((float(start), float(end), label))
    return sorted(intervals, key=lambda item: item[0])


def marker_time(markers: pd.DataFrame, marker_name: str, start_ms: float) -> float | None:
    """Retourne le premier timestamp relatif associe a un marker."""
    if not {"utcTime", "eventMarker"}.issubset(markers.columns):
        return None
    rows = markers[markers["eventMarker"].eq(marker_name)].copy()
    times = to_relative_seconds(rows["utcTime"], start_ms).dropna()
    if times.empty:
        return None
    return float(times.iloc[0])


def session_raw_end_seconds(session_dir: Path, markers: pd.DataFrame, start_ms: float) -> float | None:
    """Determine la fin de la session brute avec end_session ou les derniers timestamps capteurs."""
    end_from_marker = marker_time(markers, "end_session", start_ms)
    if end_from_marker is not None:
        return end_from_marker

    ends: list[float] = []
    for filename in [*FILE_COLS.keys(), "chest_raw_breathing.csv"]:
        path = session_dir / filename
        if not path.exists():
            continue
        try:
            timestamps = pd.read_csv(path, usecols=["timestamp"])["timestamp"]
        except (ValueError, FileNotFoundError, pd.errors.EmptyDataError):
            continue
        relative_time = to_relative_seconds(timestamps, start_ms).dropna()
        if not relative_time.empty:
            ends.append(float(relative_time.max()))
    return max(ends) if ends else None


def assign_labels(df: pd.DataFrame, intervals: list[tuple[float, float, int]]) -> pd.DataFrame:
    """Ajoute la colonne label selon les intervalles experimentaux."""
    df["label"] = np.nan
    for start, end, label in intervals:
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df.loc[mask, "label"] = label
    return df


def load_v2(participant: str, session: str) -> tuple[pd.DataFrame, list[str]]:
    """Charge une session depuis data/01_raw/<participant>/<session>/."""
    warnings: list[str] = []
    session_dir = RAW_DIR / participant / session
    if not session_dir.exists():
        return pd.DataFrame(), [f"Dossier V2 introuvable: {session_dir}"]

    start_ms = session_start_ms(session_dir)
    if start_ms is None:
        return pd.DataFrame(), [f"Aucun timestamp exploitable dans {session_dir}"]

    frames: list[pd.DataFrame] = []
    for filename, rename_map in FILE_COLS.items():
        frame, warning = read_sensor_file(session_dir / filename, rename_map, start_ms)
        if warning:
            warnings.append(warning)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(), warnings + ["Aucun capteur principal charge."]

    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            frame.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=1,
        )

    breathing, warning = compute_breathing_rpm(session_dir, merged["timestamp"], start_ms)
    if warning:
        warnings.append(warning)
    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        breathing.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=1,
    )

    age, gender = load_metadata(participant)
    merged["age"] = age
    merged["gender"] = gender
    merged["participant"] = participant
    merged["session"] = session
    merged = assign_labels(merged, label_intervals(session_dir, start_ms))

    for signal in SIGNALS:
        if signal not in merged.columns:
            merged[signal] = np.nan
    merged = merged.sort_values("timestamp")
    merged.attrs["sampling_rates"] = session_sampling_rates(session_dir, start_ms)
    return merged, warnings


@lru_cache(maxsize=1)
def v2_global_phase_percentages() -> dict[str, dict[str, float]]:
    """Calcule les pourcentages de duree baseline/activity/fatigue dans tout data/01_raw."""
    totals = {phase: 0.0 for phase in LABEL_MAP}
    if not RAW_DIR.exists():
        return {phase: {"seconds": 0.0, "percent": 0.0} for phase in LABEL_MAP}

    for markers_path in RAW_DIR.glob("[0-9][0-9]/[0-9][0-9]/exp_markers.csv"):
        session_dir = markers_path.parent
        start_ms = session_start_ms(session_dir)
        if start_ms is None:
            continue
        for start, end, label in label_intervals(session_dir, start_ms):
            if end > start:
                phase = LABELS[label]["name"]
                totals[phase] += end - start

    total_seconds = sum(totals.values())
    if total_seconds <= 0:
        return {phase: {"seconds": 0.0, "percent": 0.0} for phase in LABEL_MAP}
    return {
        phase: {"seconds": seconds, "percent": seconds / total_seconds * 100.0}
        for phase, seconds in totals.items()
    }


@lru_cache(maxsize=1)
def v2_global_training_usage() -> dict[str, float]:
    """Calcule les parts globales V2 utilisees et non utilisees par l'entrainement."""
    used_seconds = 0.0
    unused_seconds = 0.0
    total_seconds = 0.0
    sessions_count = 0

    if not RAW_DIR.exists():
        return {
            "total_seconds": 0.0,
            "used_seconds": 0.0,
            "unused_seconds": 0.0,
            "used_percent": 0.0,
            "unused_percent": 0.0,
            "sessions_count": 0,
        }

    for markers_path in RAW_DIR.glob("[0-9][0-9]/[0-9][0-9]/exp_markers.csv"):
        session_dir = markers_path.parent
        start_ms = session_start_ms(session_dir)
        if start_ms is None:
            continue

        markers = pd.read_csv(markers_path)
        start_baseline = marker_time(markers, "start_baseline", start_ms)
        end_baseline = marker_time(markers, "end_baseline", start_ms)
        start_activity = marker_time(markers, "start_activity", start_ms)
        end_activity = marker_time(markers, "end_activity", start_ms)
        start_fatigue = marker_time(markers, "start_fatigue", start_ms)
        end_fatigue = marker_time(markers, "end_fatigue", start_ms)
        session_end = session_raw_end_seconds(session_dir, markers, start_ms)

        if (
            start_baseline is None
            or end_baseline is None
            or start_activity is None
            or end_activity is None
            or start_fatigue is None
            or end_fatigue is None
            or session_end is None
        ):
            continue

        session_total = max(0.0, session_end - start_baseline)
        session_used = (
            max(0.0, end_baseline - start_baseline)
            + max(0.0, end_activity - start_activity)
            + max(0.0, end_fatigue - start_fatigue)
        )
        session_unused = (
            max(0.0, start_activity - end_baseline)
            + max(0.0, start_fatigue - end_activity)
            + max(0.0, session_end - end_fatigue)
        )

        total_seconds += session_total
        used_seconds += session_used
        unused_seconds += session_unused
        sessions_count += 1

    used_percent = used_seconds / total_seconds * 100.0 if total_seconds > 0 else 0.0
    unused_percent = unused_seconds / total_seconds * 100.0 if total_seconds > 0 else 0.0
    return {
        "total_seconds": total_seconds,
        "used_seconds": used_seconds,
        "unused_seconds": unused_seconds,
        "used_percent": used_percent,
        "unused_percent": unused_percent,
        "sessions_count": sessions_count,
    }


def phase_summary_component(version: str | list[str]) -> html.Div:
    """Prepare le panneau de repartition globale des phases V2."""
    if not version_has_v2(version):
        return html.Div("La repartition globale est calculee pour la Version 2 brute.", className="muted-text")

    percentages = v2_global_phase_percentages()
    cards = []
    for phase in ["baseline", "activity", "fatigue"]:
        stats = percentages[phase]
        cards.append(
            html.Div(
                [
                    html.Div(phase, className="phase-name"),
                    html.Div(f"{stats['percent']:.1f} %", className="phase-value"),
                    html.Div(f"{stats['seconds'] / 60.0:.1f} min", className="phase-duration"),
                ],
                className=f"phase-card phase-{phase}",
            )
        )
    return html.Div(cards, className="phase-grid")


def unused_summary_component(version: str | list[str]) -> html.Div:
    """Prepare le resume global des donnees non utilisees par le modele."""
    if not version_has_v2(version):
        return html.Div("Selectionner la Version 2 pour afficher le pourcentage de data brute non utilisee.", className="muted-text")

    stats = v2_global_training_usage()
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Data utilisee pour entrainement", className="phase-name"),
                    html.Div(f"{stats['used_percent']:.1f} %", className="phase-value"),
                    html.Div(f"{stats['used_seconds'] / 60.0:.1f} min", className="phase-duration"),
                ],
                className="phase-card phase-used",
            ),
            html.Div(
                [
                    html.Div("Data non utilisee par le modele", className="phase-name"),
                    html.Div(f"{stats['unused_percent']:.1f} %", className="phase-value"),
                    html.Div(
                        "gaps: baseline->activity, activity->fatigue, fatigue->fin",
                        className="phase-duration",
                    ),
                ],
                className="phase-card phase-unused",
            ),
            html.Div(
                [
                    html.Div("Total brut V2 analyse", className="phase-name"),
                    html.Div(f"{stats['total_seconds'] / 60.0:.1f} min", className="phase-value"),
                    html.Div(f"{int(stats['sessions_count'])} sessions", className="phase-duration"),
                ],
                className="phase-card",
            ),
        ],
        className="phase-grid",
    )


def version_has_v2(version_selection: str | list[str]) -> bool:
    """Indique si la selection courante contient la Version 2."""
    if isinstance(version_selection, str):
        return version_selection == "v2"
    return "v2" in (version_selection or [])


def version_has_v1(version_selection: str | list[str]) -> bool:
    """Indique si la selection courante contient la Version 1."""
    if isinstance(version_selection, str):
        return version_selection == "v1"
    return "v1" in (version_selection or [])


def normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalise chaque signal en min-max; les constantes restent tracables."""
    normalized = df.copy()
    for column in columns:
        series = cast(pd.Series, pd.to_numeric(normalized[column], errors="coerce"))
        minimum = float(series.min(skipna=True))
        maximum = float(series.max(skipna=True))
        if pd.isna(minimum) or pd.isna(maximum):
            normalized[column] = np.nan
        elif maximum == minimum:
            normalized[column] = 0.5
        else:
            normalized[column] = (series - minimum) / (maximum - minimum)
    return normalized


def add_label_background(fig: go.Figure, df: pd.DataFrame) -> None:
    """Colore le fond du graphe selon les labels disponibles."""
    if "label" not in df.columns or df.empty:
        return

    labels = cast(pd.Series, pd.to_numeric(df["label"], errors="coerce"))
    timestamps = df["timestamp"].to_numpy()
    if cast(pd.Series, labels.dropna()).empty:
        return

    current_label = float(labels.iloc[0])
    start = timestamps[0]
    legend_done: set[int] = set()
    for idx in range(1, len(df)):
        if float(labels.iloc[idx]) != current_label:
            add_label_shape(fig, start, timestamps[idx - 1], current_label, legend_done)
            start = timestamps[idx]
            current_label = float(labels.iloc[idx])
    add_label_shape(fig, start, timestamps[-1], current_label, legend_done)


def add_label_shape(fig: go.Figure, x0: float, x1: float, label: float, legend_done: set[int]) -> None:
    """Ajoute un rectangle de fond et une entree de legende pour un label."""
    if pd.isna(label) or int(label) not in LABELS or x1 <= x0:
        return

    label_int = int(label)
    meta = LABELS[label_int]
    fig.add_vrect(x0=x0, x1=x1, fillcolor=meta["color"], line_width=0, layer="below")
    if label_int not in legend_done:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 12, "color": meta["color"].replace("0.14", "0.45").replace("0.13", "0.45")},
                name=meta["name"],
                legendgroup="labels",
                showlegend=True,
            )
        )
        legend_done.add(label_int)


def create_figure(df: pd.DataFrame, selected_signals: list[str], scale_mode: str) -> go.Figure:
    """Construit le graphe Plotly unique avec echelle brute ou normalisee."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", annotations=[{"text": "Aucune donnee a afficher", "showarrow": False}])
        return fig

    selected_signals = [signal for signal in selected_signals if signal in df.columns]
    plot_df = normalize(df, selected_signals) if scale_mode == "normalized" else df
    sampling_rates = df.attrs.get("sampling_rates", {})
    fig = go.Figure()
    add_label_background(fig, df)

    for index, signal in enumerate(selected_signals, start=1):
        yaxis = "y" if scale_mode == "normalized" or index == 1 else f"y{index}"
        unit = "0-1" if scale_mode == "normalized" else SIGNAL_UNITS.get(signal, "")
        hover_unit = f" {unit}" if unit else ""
        fig.add_trace(
            go.Scatter(
                x=plot_df["timestamp"],
                y=pd.to_numeric(plot_df[signal], errors="coerce"),
                mode="lines",
                name=signal_label(signal, sampling_rates),
                yaxis=yaxis,
                line={"width": 1.6},
                hovertemplate=f"{signal}: %{{y:.4g}}{hover_unit}<br>t=%{{x:.2f}} s<extra></extra>",
            )
        )

    y_title = "Valeur normalisee (0-1)" if scale_mode == "normalized" else signal_label(selected_signals[0], sampling_rates) if selected_signals else ""
    layout = {
        "template": "plotly_white",
        "height": 720,
        "margin": {"l": 72, "r": 72, "t": 42, "b": 64},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        "xaxis": {"title": "Temps relatif (s)", "rangeslider": {"visible": False}},
        "yaxis": {"title": y_title},
    }

    if scale_mode == "raw":
        for index, signal in enumerate(selected_signals[1:], start=2):
            side = "right" if index % 2 == 0 else "left"
            position = min(0.98, max(0.02, 1.0 - (index // 2) * 0.045)) if side == "right" else max(0.02, (index // 2) * 0.045)
            layout[f"yaxis{index}"] = {
                "title": signal_label(signal, sampling_rates),
                "overlaying": "y",
                "side": side,
                "anchor": "free",
                "position": position,
                "showgrid": False,
                "zeroline": False,
            }
    fig.update_layout(**layout)
    return fig


def normalize_with_bounds(series: pd.Series, minimum: float, maximum: float) -> pd.Series:
    """Normalise une serie avec des bornes communes pour comparer V1 et V2."""
    values = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(np.nan, index=series.index)
    if maximum == minimum:
        return pd.Series(0.5, index=series.index)
    return cast(pd.Series, (values - minimum) / (maximum - minimum))


def comparison_bounds(datasets: dict[str, pd.DataFrame], selected_signals: list[str]) -> dict[str, tuple[float, float]]:
    """Calcule les bornes min-max communes par signal pour la comparaison normalisee."""
    bounds: dict[str, tuple[float, float]] = {}
    for signal in selected_signals:
        values = []
        for df in datasets.values():
            if signal in df.columns and not df.empty:
                values.append(pd.to_numeric(df[signal], errors="coerce"))
        if not values:
            bounds[signal] = (np.nan, np.nan)
            continue
        merged = pd.concat(values, ignore_index=True)
        bounds[signal] = (merged.min(skipna=True), merged.max(skipna=True))
    return bounds


def create_comparison_figure(datasets: dict[str, pd.DataFrame], selected_signals: list[str], scale_mode: str) -> go.Figure:
    """Construit un graphe comparatif superposant V1 et V2 pour les memes signaux."""
    datasets = {version: df for version, df in datasets.items() if not df.empty}
    if not datasets:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", annotations=[{"text": "Aucune donnee a afficher", "showarrow": False}])
        return fig

    selected_signals = [
        signal
        for signal in selected_signals
        if any(signal in df.columns for df in datasets.values())
    ]
    fig = go.Figure()
    background_df = next(iter(datasets.values()))
    add_label_background(fig, background_df)
    bounds = comparison_bounds(datasets, selected_signals) if scale_mode == "normalized" else {}
    signal_axis = {signal: index for index, signal in enumerate(selected_signals, start=1)}
    version_names = {"v1": "V1", "v2": "V2"}
    version_styles = {
        "v1": {"dash": "solid", "width": 1.8},
        "v2": {"dash": "dot", "width": 1.9},
    }

    for signal in selected_signals:
        axis_index = signal_axis[signal]
        yaxis = "y" if scale_mode == "normalized" or axis_index == 1 else f"y{axis_index}"
        unit = "0-1" if scale_mode == "normalized" else SIGNAL_UNITS.get(signal, "")
        hover_unit = f" {unit}" if unit else ""
        minimum, maximum = bounds.get(signal, (np.nan, np.nan))
        for version, df in datasets.items():
            if signal not in df.columns:
                continue
            sampling_rates = df.attrs.get("sampling_rates", {})
            y_values = (
                normalize_with_bounds(df[signal], minimum, maximum)
                if scale_mode == "normalized"
                else pd.to_numeric(df[signal], errors="coerce")
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=y_values,
                    mode="lines",
                    name=f"{version_names.get(version, version)} - {signal_label(signal, sampling_rates)}",
                    yaxis=yaxis,
                    line=version_styles.get(version, {"width": 1.6}),
                    hovertemplate=f"{version_names.get(version, version)} {signal}: %{{y:.4g}}{hover_unit}<br>t=%{{x:.2f}} s<extra></extra>",
                )
            )

    first_signal = selected_signals[0] if selected_signals else ""
    first_rates = next(iter(datasets.values())).attrs.get("sampling_rates", {})
    y_title = "Valeur normalisee commune (0-1)" if scale_mode == "normalized" else signal_label(first_signal, first_rates)
    layout = {
        "template": "plotly_white",
        "height": 720,
        "margin": {"l": 72, "r": 72, "t": 42, "b": 64},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        "xaxis": {"title": "Temps relatif (s)", "rangeslider": {"visible": False}},
        "yaxis": {"title": y_title},
    }

    if scale_mode == "raw":
        first_dataset = next(iter(datasets.values()))
        first_rates = first_dataset.attrs.get("sampling_rates", {})
        for signal in selected_signals[1:]:
            axis_index = signal_axis[signal]
            side = "right" if axis_index % 2 == 0 else "left"
            position = min(0.98, max(0.02, 1.0 - (axis_index // 2) * 0.045)) if side == "right" else max(0.02, (axis_index // 2) * 0.045)
            layout[f"yaxis{axis_index}"] = {
                "title": signal_label(signal, first_rates),
                "overlaying": "y",
                "side": side,
                "anchor": "free",
                "position": position,
                "showgrid": False,
                "zeroline": False,
            }
    fig.update_layout(**layout)
    return fig


app = Dash(__name__)
app.title = "Fatigue Wearables Dashboard"

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Visualisation fatigue wearables", className="app-title"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Versions des donnees"),
                                dcc.Checklist(
                                    id="data-version",
                                    options=[
                                        {"label": "Version 1 (CSV unique)", "value": "v1"},
                                        {"label": "Version 2 (dossiers)", "value": "v2"},
                                    ],
                                    value=["v1"],
                                    inline=True,
                                    className="radio-row",
                                ),
                            ],
                            className="control-block wide",
                        ),
                        html.Div(
                            [
                                html.Label("Participant"),
                                dcc.Dropdown(
                                    id="participant",
                                    options=[{"label": f"{idx:02d}", "value": f"{idx:02d}"} for idx in range(1, 13)],
                                    value="01",
                                    clearable=False,
                                ),
                            ],
                            className="control-block",
                        ),
                        html.Div(
                            [
                                html.Label("Session"),
                                dcc.Dropdown(
                                    id="session",
                                    options=[{"label": f"{idx:02d}", "value": f"{idx:02d}"} for idx in range(1, 4)],
                                    value="01",
                                    clearable=False,
                                ),
                            ],
                            className="control-block",
                        ),
                        html.Div(
                            [
                                html.Label("Echelle Y"),
                                dcc.Checklist(
                                    id="scale-switch",
                                    options=[{"label": "Echelle normalisee", "value": "normalized"}],
                                    value=[],
                                    className="switch-like",
                                ),
                            ],
                            className="control-block",
                        ),
                    ],
                    className="controls-grid",
                ),
                html.Div(
                    [
                        html.Label("Signaux"),
                        dcc.Checklist(
                            id="signals",
                            options=[{"label": signal, "value": signal} for signal in SIGNALS],
                            value=["acc_x", "acc_y", "acc_z", "eda", "wrist_hr", "temp", "breathing_rpm"],
                            inline=True,
                            className="signals-list",
                        ),
                    ],
                    className="signals-panel",
                ),
            ],
            className="top-panel",
        ),
        html.Div(
            [
                html.Div("Repartition globale des phases - dataset brut V2", className="panel-title"),
                html.Div(id="phase-summary"),
            ],
            className="summary-panel",
        ),
        html.Div(id="warnings", className="warning-box"),
        dcc.Loading(dcc.Graph(id="main-graph", config={"displaylogo": False, "scrollZoom": True}), type="default"),
        html.Div(
            [
                html.Div("Pourcentage global de data brute non utilisee par le modele", className="panel-title"),
                html.Div(id="unused-summary"),
            ],
            className="summary-panel bottom-panel",
        ),
    ],
    className="dashboard-container",
)

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background: #f5f7fb; color: #1f2937; }
            .dashboard-container { padding: 24px; max-width: 1500px; }
            .top-panel { background: white; border: 1px solid #d9e2ef; border-radius: 8px; padding: 18px; margin-bottom: 14px; }
            .app-title { font-size: 1.55rem; margin: 0 0 16px; font-weight: 700; }
            .controls-grid { display: flex; flex-wrap: wrap; gap: 14px; align-items: end; }
            .control-block { min-width: 180px; flex: 1 1 180px; }
            .control-block.wide { flex: 2 1 360px; }
            .control-block label, .signals-panel label { font-weight: 600; margin-bottom: 6px; display: block; }
            .radio-row label, .signals-list label { margin-right: 16px; font-weight: 400; }
            .switch-like label { font-weight: 400; margin: 0; }
            .signals-panel { margin-top: 16px; }
            .signals-list { display: flex; flex-wrap: wrap; gap: 8px 12px; }
            .summary-panel { background: white; border: 1px solid #d9e2ef; border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; }
            .panel-title { font-weight: 700; margin-bottom: 10px; }
            .phase-grid { display: grid; grid-template-columns: repeat(3, minmax(140px, 1fr)); gap: 10px; }
            .phase-card { border-radius: 8px; padding: 12px; border: 1px solid #d9e2ef; }
            .phase-baseline { background: rgba(65, 105, 225, 0.10); }
            .phase-activity { background: rgba(46, 160, 67, 0.10); }
            .phase-fatigue { background: rgba(220, 53, 69, 0.10); }
            .phase-used { background: rgba(46, 160, 67, 0.10); }
            .phase-unused { background: rgba(245, 158, 11, 0.14); }
            .phase-name { font-weight: 700; }
            .phase-value { font-size: 1.4rem; font-weight: 800; margin-top: 4px; }
            .phase-duration, .muted-text { color: #526173; font-size: 0.92rem; }
            .warning-box { display: none; background: #fff7e6; border: 1px solid #f4c36a; color: #5f4300; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; }
            .warning-box:not(:empty) { display: block; }
            .warning-box ul { margin: 0; padding-left: 18px; }
            #main-graph { background: white; border: 1px solid #d9e2ef; border-radius: 8px; padding: 8px; }
            @media (max-width: 720px) {
                .dashboard-container { padding: 12px; }
                .top-panel { padding: 14px; }
                .controls-grid { display: block; }
                .control-block { margin-bottom: 12px; }
                .phase-grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


@app.callback(
    Output("main-graph", "figure"),
    Output("warnings", "children"),
    Output("phase-summary", "children"),
    Output("unused-summary", "children"),
    Input("data-version", "value"),
    Input("participant", "value"),
    Input("session", "value"),
    Input("signals", "value"),
    Input("scale-switch", "value"),
)
def update_graph(version: list[str], participant: str, session: str, signals: list[str], scale_values: list[str]):
    participant = participant.zfill(2)
    session = session.zfill(2)
    selected_versions = version or ["v1"]
    datasets: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    if "v1" in selected_versions:
        df_v1, warnings_v1 = load_v1(participant, session)
        datasets["v1"] = df_v1
        warnings.extend([f"V1: {warning}" for warning in warnings_v1])
    if "v2" in selected_versions:
        df_v2, warnings_v2 = load_v2(participant, session)
        datasets["v2"] = df_v2
        warnings.extend([f"V2: {warning}" for warning in warnings_v2])

    scale_mode = "normalized" if scale_values and "normalized" in scale_values else "raw"
    fig = create_comparison_figure(datasets, signals or [], scale_mode)
    warning_text = html.Ul([html.Li(warning) for warning in warnings]) if warnings else ""
    return fig, warning_text, phase_summary_component(selected_versions), unused_summary_component(selected_versions)


if __name__ == "__main__":
    # use_reloader=False évite que le débogueur VS Code (debugpy) n'intercepte SystemExit: 3 lors du rechargement
    app.run(debug=True, use_reloader=False)

