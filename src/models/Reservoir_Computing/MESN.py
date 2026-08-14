import gc
import json
import random
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from reservoirpy.nodes import Reservoir, Ridge
from scipy.signal import butter, filtfilt

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import *

import onnx
import onnx.helper
import onnx.numpy_helper
import onnxruntime as ort

from utils.apply_smote import resample_windows_smote
from models.semi_supervised import add_pseudo_labels

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 1. CONSTANTES
# ============================================================================
MODEL_NAME = "MESN"
TARGET_FREQ_CAP_HZ = 1.0
LABEL_MAPPING = {"baseline": 0, "activity": 1, "pre_fatigue": 2, "fatigue": 3}
TARGET_NAMES = ["baseline", "activity", "pre_fatigue", "fatigue"]

DEFAULT_PARAMS = MODEL_PARAMS.get("MESN", {})

SENSOR_SPECS = {
    "acc": {
        "file": "wrist_acc.csv",
        "cols": {"ax": "acc_x", "ay": "acc_y", "az": "acc_z"},
        "freq": 32.0,
        "log1p": [],
    },
    "ambient": {
        "file": "ambient_grandeur.csv",
        "cols": {"temperature_amb": "temp_amb", "humidite_amb": "hum_amb"},
        "freq": 4.0,
        "log1p": [],
    },
    "hr": {
        "file": "wrist_hr.csv",
        "cols": {"hr": "wrist_hr"},
        "freq": 1.0,
        "log1p": [],
    },
    "ibi": {
        "file": "wrist_ibi.csv",
        "cols": {"duration": "ibi"},
        "freq": 0.65,
        "log1p": [],
    },
    "temp": {
        "file": "wrist_skin_temperature.csv",
        "cols": {"temp": "temp"},
        "freq": 4.0,
        "log1p": [],
    },
    "breathing": {
        "file": "chest_raw_breathing.csv",
        "cols": {"breathing_waveform": "breathing_waveform"},
        "freq": 25.0,
        "log1p": [],
    },
}

SENSOR_SPECS_CSV = {
    **{k: v for k, v in SENSOR_SPECS.items() if k != "breathing"},
    "breathing": {
        "file": None,  # non utilise dans ce chemin (donnee deja dans le CSV)
        "cols": {"breathing_rpm": "breathing_rpm"},
        "freq": 0.5,  # taux respiratoire calcule, pas la waveform brute a 25Hz
        "log1p": [],
    },
}


@dataclass
class SessionData:
    participant: int
    session: int
    intervals: list
    sensors: dict


# ============================================================================
# 2. LECTURE RAW SANS SYNCHRONISATION 4 HZ
# ============================================================================
def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _window_ms(participant):
    config = WINDOW_CONFIGS.get(participant, WINDOW_CONFIGS["default"])
    window_ms = int(config["window_size"] / 4 * 1000)
    step_ms = int(config["step_size"] / 4 * 1000)
    return window_ms, step_ms


def _find_sensor_files(session_dir, filename):
    stem = filename.replace(".csv", "")
    return sorted(session_dir.glob(f"{stem}*.csv"))


def _get_label_intervals(markers_path):
    df = pd.read_csv(markers_path)
    df.columns = df.columns.str.strip()

    intervals = []
    active = {}
    labels = ["baseline", "activity", "fatigue"]

    for _, row in df.iterrows():
        ts = float(row["utcTime"])
        event = str(row["eventMarker"]).lower()

        for label in labels:
            if f"start_{label}" in event:
                active[label] = ts
            elif f"end_{label}" in event and label in active:
                intervals.append((active.pop(label), ts, label))

    if not intervals:
        return [], None

    t0 = intervals[0][0]
    return [(start - t0, end - t0, label) for start, end, label in intervals], t0


def _augment_intervals(intervals):
    augmented = list(intervals)
    bounds = {}
    for start, end, label in intervals:
        if label == "baseline":
            bounds["baseline_end"] = end
        elif label == "activity":
            bounds["activity_start"] = start
            bounds["activity_end"] = end
        elif label == "fatigue":
            bounds["fatigue_start"] = start

    activity_end = bounds.get("activity_end")
    fatigue_start = bounds.get("fatigue_start")
    if activity_end is not None and fatigue_start is not None and activity_end < fatigue_start:
        augmented.append((activity_end, fatigue_start, "pre_fatigue"))

    baseline_end = bounds.get("baseline_end")
    activity_start = bounds.get("activity_start")
    if baseline_end is not None and activity_start is not None and baseline_end < activity_start:
        augmented.append((baseline_end, activity_start, "unlabeled"))

    return augmented


def _read_sensor(session_dir, sensor_name, t0):
    spec = SENSOR_SPECS[sensor_name]
    files = _find_sensor_files(session_dir, spec["file"])
    if not files:
        return None

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    time_cols = [col for col in df.columns if "time" in col]
    if not time_cols:
        return None

    df = df.rename(columns={time_cols[0]: COL_TIMESTAMP})
    cols = {COL_TIMESTAMP: COL_TIMESTAMP}
    for raw_col, clean_col in spec["cols"].items():
        if raw_col in df.columns:
            cols[raw_col] = clean_col

    if len(cols) == 1:
        return None

    df = df[list(cols.keys())].rename(columns=cols)
    value_cols = [col for col in df.columns if col != COL_TIMESTAMP]
    df[COL_TIMESTAMP] = pd.to_numeric(df[COL_TIMESTAMP], errors="coerce") - float(t0)
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[COL_TIMESTAMP])
    df = df.dropna(subset=value_cols, how="all")
    df = df.sort_values(COL_TIMESTAMP).drop_duplicates(subset=[COL_TIMESTAMP])
    if df.empty:
        return None

    for col in value_cols:
        missing_col = f"{col}_missing"
        df[missing_col] = df[col].isna().astype(np.float32)
        limit = max(1, int(round(spec["freq"] * 2)))
        df[col] = df[col].interpolate(limit=limit, limit_direction="both")
        df[col] = df[col].ffill().bfill()
        if col in spec["log1p"]:
            df[col] = np.log1p(df[col].clip(lower=0))

    missing_cols = [f"{col}_missing" for col in value_cols]
    return df[[COL_TIMESTAMP] + value_cols + missing_cols].reset_index(drop=True)


def _filter_breathing(df):
    if df is None or "breathing_waveform" not in df.columns or len(df) < 80:
        return df

    x = df["breathing_waveform"].astype(np.float64).to_numpy()
    x = x - np.nanmean(x)
    try:
        b, a = butter(2, [0.05, 0.80], btype="band", fs=25.0)
        df["breathing_waveform"] = filtfilt(b, a, x).astype(np.float32)
    except ValueError:
        df["breathing_waveform"] = x.astype(np.float32)
    return df


def load_raw_sessions(raw_dir=DATA_RAW):
    sessions = []

    for p_dir in sorted(raw_dir.iterdir()):
        if not p_dir.is_dir():
            continue
        for s_dir in sorted(p_dir.iterdir()):
            if not s_dir.is_dir():
                continue

            marker_path = s_dir / "exp_markers.csv"
            if not marker_path.exists():
                continue

            intervals, t0 = _get_label_intervals(marker_path)
            if not intervals or t0 is None:
                continue
            intervals = _augment_intervals(intervals)

            sensors = {}
            for sensor_name in SENSOR_SPECS:
                df_sensor = _read_sensor(s_dir, sensor_name, t0)
                if sensor_name == "breathing":
                    df_sensor = _filter_breathing(df_sensor)
                if df_sensor is not None and not df_sensor.empty:
                    sensors[sensor_name] = df_sensor

            if sensors:
                sessions.append(
                    SessionData(
                        participant=_as_int(p_dir.name),
                        session=_as_int(s_dir.name),
                        intervals=intervals,
                        sensors=sensors,
                    )
                )

    return sessions


LABEL_INT_TO_NAME = {0: "baseline", 1: "activity", 2: "pre_fatigue", 3: "fatigue", -1: "unlabeled"}


def _build_intervals_from_label_column(group_df):
    intervals = []
    labels = group_df[COL_LABEL].to_numpy()
    timestamps = group_df[COL_TIMESTAMP].to_numpy()

    run_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[run_start]:
            label_name = LABEL_INT_TO_NAME.get(int(labels[run_start]))
            if label_name is not None:
                intervals.append(
                    (float(timestamps[run_start]), float(timestamps[i - 1]), label_name)
                )
            run_start = i

    return intervals


def load_sessions_from_csv(csv_path, sensor_specs=None):
    if sensor_specs is None:
        sensor_specs = SENSOR_SPECS
    df = pd.read_csv(csv_path)
    sessions = []

    for (participant, session_id), group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        group = group.sort_values(COL_TIMESTAMP).reset_index(drop=True)

        intervals = _build_intervals_from_label_column(group)
        if not intervals:
            continue

        sensors = {}
        for sensor_name, spec in sensor_specs.items():
            value_cols = list(spec["cols"].values())
            available = [c for c in value_cols if c in group.columns]
            if not available:
                continue

            sub = group[[COL_TIMESTAMP] + available].dropna(subset=available, how="all")
            if sub.empty:
                continue

            for col in available:
                sub[f"{col}_missing"] = 0.0

            sensors[sensor_name] = sub.reset_index(drop=True)

        if sensors:
            sessions.append(
                SessionData(
                    participant=_as_int(participant),
                    session=_as_int(session_id),
                    intervals=intervals,
                    sensors=sensors,
                )
            )

    print(f"  Sessions chargees depuis CSV : {len(sessions)}")
    return sessions


# ============================================================================
# 3. RESERVOIRS ET FEATURES MESN
# ============================================================================
def _free_memory(*objects):
    for obj in objects:
        del obj
    gc.collect()


def _build_reservoir(sensor_name, params, input_dim):
    sensor_params = params["reservoirs"][sensor_name]
    sparsity = float(sensor_params["sparsity"])
    rc_connectivity = max(0.0, min(1.0, 1.0 - sparsity))

    node = Reservoir(
        units=int(sensor_params["n_reservoir"]),
        lr=float(sensor_params["leak_rate"]),
        sr=float(sensor_params["spectral_radius"]),
        input_scaling=float(sensor_params["input_scaling"]),
        input_connectivity=1.0,
        rc_connectivity=rc_connectivity,
        input_dim=int(input_dim),
        dtype=np.float32,
        seed=int(params.get("random_state", 42)),
        name=f"{sensor_name}_reservoir",
    )

    return {
        "node": node,
        "params": dict(sensor_params),
        "input_dim": int(input_dim),
        "sensor": sensor_name,
    }


def _reservoir_states(window, reservoir, washout):
    node = reservoir["node"]
    if node.initialized:
        node.reset()

    states = node.run(window.astype(np.float32))
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states.reshape(1, -1)

    return states[washout:] if washout < len(states) else states[-1:]


def _summarize_states(states, summary):
    last = states[-1]
    if summary == "last":
        return last.astype(np.float32)
    if summary == "last_mean":
        return np.concatenate([last, states.mean(axis=0)]).astype(np.float32)
    if summary == "last_mean_std":
        return np.concatenate([last, states.mean(axis=0), states.std(axis=0)]).astype(np.float32)
    raise ValueError(f"state_summary inconnu: {summary}")


def _feature_size(params, sensor_name):
    units = int(params["reservoirs"][sensor_name]["n_reservoir"])
    summary = params.get("state_summary", "last_mean_std")
    if summary == "last":
        return units
    if summary == "last_mean":
        return units * 2
    if summary == "last_mean_std":
        return units * 3
    raise ValueError(f"state_summary inconnu: {summary}")


def _make_scalers(train_sessions, sensor_specs=None):
    if sensor_specs is None:
        sensor_specs = SENSOR_SPECS
    scalers = {}
    for sensor_name, spec in sensor_specs.items():
        value_cols = list(spec["cols"].values())
        arrays = []
        for session in train_sessions:
            df = session.sensors.get(sensor_name)
            if df is not None and all(col in df.columns for col in value_cols):
                arrays.append(df[value_cols].to_numpy(dtype=np.float32))

        if arrays:
            x = np.vstack(arrays)
            scaler = RobustScaler()
            scaler.fit(x)
            scalers[sensor_name] = scaler

    return scalers


def _build_reservoirs(params, scalers, sensor_specs=None):
    if sensor_specs is None:
        sensor_specs = SENSOR_SPECS
    reservoirs = {}
    for sensor_name, scaler in scalers.items():
        reservoirs[sensor_name] = _build_reservoir(
            sensor_name,
            params,
            input_dim=len(sensor_specs[sensor_name]["cols"]) * 2,
        )
    return reservoirs


def _label_at(intervals, start_ms, end_ms):
    overlaps = []
    for s, e, label in intervals:
        overlap = max(0.0, min(end_ms, e) - max(start_ms, s))
        if overlap > 0:
            overlaps.append((overlap, label))
    if not overlaps:
        return None
    return max(overlaps, key=lambda item: item[0])[1]


def _session_time_bounds(session):
    starts = [df[COL_TIMESTAMP].min() for df in session.sensors.values() if not df.empty]
    ends = [df[COL_TIMESTAMP].max() for df in session.sensors.values() if not df.empty]
    if not starts or not ends:
        return None, None
    label_start = min(s for s, _, _ in session.intervals)
    label_end = max(e for _, e, _ in session.intervals)
    return max(min(starts), label_start), min(max(ends), label_end)


def _sensor_window(df, start_ms, end_ms, value_cols):
    mask = (df[COL_TIMESTAMP] >= start_ms) & (df[COL_TIMESTAMP] < end_ms)
    return df.loc[mask, value_cols].to_numpy(dtype=np.float32)


def _downsample_factor(sensor_name, sensor_specs=None):
    if sensor_specs is None:
        sensor_specs = SENSOR_SPECS
    freq = float(sensor_specs[sensor_name]["freq"])
    return max(1, int(freq // TARGET_FREQ_CAP_HZ))


def _effective_freq(sensor_name, sensor_specs=None):
    if sensor_specs is None:
        sensor_specs = SENSOR_SPECS
    freq = float(sensor_specs[sensor_name]["freq"])
    return freq / _downsample_factor(sensor_name, sensor_specs=sensor_specs)


def _downsample_block_average(x, factor):
    """
    Sous-echantillonne par moyennage de blocs consecutifs. Le dernier bloc
    incomplet est tronque pour garder un filtrage simple et deterministe, et
    eviter un simple stride qui introduirait davantage d'aliasing.
    """
    if factor <= 1 or len(x) < factor:
        return x
    n_blocks = len(x) // factor
    trimmed = x[: n_blocks * factor]
    return trimmed.reshape(n_blocks, factor, x.shape[1]).mean(axis=1)


def _fix_sensor_window_length(x_raw, n_steps, n_features):
    if len(x_raw) >= n_steps:
        return x_raw[-n_steps:]
    if len(x_raw) == 0:
        x_fixed = np.zeros((n_steps, n_features * 2), dtype=np.float32)
        x_fixed[:, n_features:] = 1.0
        return x_fixed

    pad = np.zeros((n_steps - len(x_raw), n_features * 2), dtype=np.float32)
    pad[:, n_features:] = 1.0
    return np.concatenate([pad, x_raw], axis=0).astype(np.float32)


def _sensor_inputs(df, sensor_name, start_ms, end_ms, scalers, sensor_specs=None):
    if sensor_specs is None:
        sensor_specs = SENSOR_SPECS
    value_cols = list(sensor_specs[sensor_name]["cols"].values())
    missing_cols = [f"{col}_missing" for col in value_cols]
    n_features = len(value_cols)
    x_values = _sensor_window(df, start_ms, end_ms, value_cols)
    x_missing = _sensor_window(df, start_ms, end_ms, missing_cols)
    factor = _downsample_factor(sensor_name, sensor_specs=sensor_specs)
    x_values = _downsample_block_average(x_values, factor)
    x_missing = _downsample_block_average(x_missing, factor)
    x_raw = np.concatenate([x_values, x_missing.astype(np.float32)], axis=1).astype(np.float32)

    # Meme convention que l'entree ONNX statique : downsampling, puis
    # troncature/padding brut a la taille nominale, puis RobustScaler.
    n_steps = _nominal_sensor_steps(sensor_specs, end_ms - start_ms)[sensor_name]
    x_raw = _fix_sensor_window_length(x_raw, n_steps, n_features)
    x_values = scalers[sensor_name].transform(x_raw[:, :n_features]).astype(np.float32)
    x_missing = x_raw[:, n_features:].astype(np.float32)
    return np.concatenate([x_values, x_missing], axis=1)


def _has_min_coverage(sensor_name, x, window_ms, params, sensor_specs=None):
    if sensor_specs is None:
        sensor_specs = SENSOR_SPECS
    expected = sensor_specs[sensor_name]["freq"] * (window_ms / 1000)
    min_samples = max(2, int(round(expected * float(params.get("min_coverage", 0.35)))))
    if sensor_specs[sensor_name]["freq"] < 1:
        min_samples = max(2, min(min_samples, 8))
    return len(x) >= min_samples


def transform_session_windows(session, reservoirs, scalers, params, sensor_specs=None):
    if sensor_specs is None:
        sensor_specs = SENSOR_SPECS
    window_ms, step_ms = _window_ms(session.participant)
    start_bound, end_bound = _session_time_bounds(session)
    if start_bound is None or end_bound is None or end_bound - start_bound < window_ms:
        return (
            np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int32), [],
            np.empty((0, 0), dtype=np.float32), [],
        )

    x_all, y_all, meta = [], [], []
    x_unlabeled, meta_unlabeled = [], []
    n_pre_fatigue, n_unlabeled = 0, 0
    washout = int(params.get("washout", 0))
    summary = params.get("state_summary", "last_mean_std")
    sensor_names = list(reservoirs.keys())

    start = float(start_bound)
    while start + window_ms <= end_bound:
        end = start + window_ms
        label = _label_at(session.intervals, start, end)
        if label is None:
            start += step_ms
            continue

        features = []
        valid = True
        for sensor_name in sensor_names:
            df_sensor = session.sensors.get(sensor_name)
            value_cols = list(sensor_specs[sensor_name]["cols"].values())
            missing_cols = [f"{col}_missing" for col in value_cols]
            required_cols = value_cols + missing_cols
            if df_sensor is None or not all(col in df_sensor.columns for col in required_cols):
                valid = False
                break

            x_sensor = _sensor_window(df_sensor, start, end, value_cols)
            if not _has_min_coverage(sensor_name, x_sensor, window_ms, params, sensor_specs=sensor_specs):
                valid = False
                break

            x_input = _sensor_inputs(df_sensor, sensor_name, start, end, scalers, sensor_specs=sensor_specs)
            states = _reservoir_states(x_input, reservoirs[sensor_name], washout)
            features.append(_summarize_states(states, summary))

        if valid:
            feature_vec = np.concatenate(features).astype(np.float32)
            meta_entry = {
                "participant": session.participant,
                "session": session.session,
                "start_ms": start,
                "end_ms": end,
                "label": label,
            }
            if label == "unlabeled":
                n_unlabeled += 1
                x_unlabeled.append(feature_vec)
                meta_unlabeled.append(meta_entry)
            else:
                if label == "pre_fatigue":
                    n_pre_fatigue += 1
                x_all.append(feature_vec)
                y_all.append(LABEL_MAPPING[label])
                meta.append(meta_entry)

        start += step_ms

    if n_pre_fatigue or n_unlabeled:
        print(
            f"  [P{session.participant} S{session.session}] "
            f"fenetres pre_fatigue: {n_pre_fatigue} | unlabeled: {n_unlabeled}"
        )

    feature_dim = sum(_feature_size(params, name) for name in sensor_names)
    if not x_all:
        x_all_result = np.empty((0, feature_dim), dtype=np.float32)
        y_all_result = np.array([], dtype=np.int32)
    else:
        x_all_result = np.asarray(x_all, dtype=np.float32)
        y_all_result = np.asarray(y_all, dtype=np.int32)

    if not x_unlabeled:
        x_unlabeled_result = np.empty((0, feature_dim), dtype=np.float32)
    else:
        x_unlabeled_result = np.asarray(x_unlabeled, dtype=np.float32)

    return x_all_result, y_all_result, meta, x_unlabeled_result, meta_unlabeled


def extract_mesn_windows(sessions, reservoirs, scalers, params, sensor_specs=None):
    xs, ys, metas = [], [], []
    xs_unlabeled, metas_unlabeled = [], []
    for session in sessions:
        x_session, y_session, meta_session, x_unl_session, meta_unl_session = transform_session_windows(
            session, reservoirs, scalers, params, sensor_specs=sensor_specs
        )
        if len(y_session) > 0:
            xs.append(x_session)
            ys.append(y_session)
            metas.extend(meta_session)
        if len(x_unl_session) > 0:
            xs_unlabeled.append(x_unl_session)
            metas_unlabeled.extend(meta_unl_session)

    feature_dim = sum(_feature_size(params, name) for name in reservoirs.keys())
    x_result = np.vstack(xs).astype(np.float32) if xs else np.empty((0, feature_dim), dtype=np.float32)
    y_result = np.concatenate(ys).astype(np.int32) if ys else np.array([], dtype=np.int32)
    x_unlabeled_result = np.vstack(xs_unlabeled).astype(np.float32) if xs_unlabeled else np.empty((0, feature_dim), dtype=np.float32)

    return x_result, y_result, metas, x_unlabeled_result, metas_unlabeled


# ============================================================================
# 4. READOUT
# ============================================================================
def build_readout(params):
    return Ridge(ridge=float(params["ridge_alpha"]), name="mesn_readout")


def _sample_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = dict(zip(classes, weights))
    return np.asarray([weight_map[y] for y in y_train], dtype=np.float32)


def _to_onehot(y, num_classes=None):
    if num_classes is None:
        num_classes = len(LABEL_MAPPING)
    y_onehot = np.zeros((len(y), num_classes), dtype=np.float32)
    y_onehot[np.arange(len(y)), y.astype(int)] = 1.0
    return y_onehot


def fit_readout(model, x_train, y_train):
    weights = np.sqrt(_sample_weights(y_train)).reshape(-1, 1)
    model.fit(x_train * weights, _to_onehot(y_train) * weights)
    return model


def predict_readout(model, x):
    scores = np.asarray(model.run(x), dtype=np.float32)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    return np.argmax(scores, axis=1).astype(np.int32)


# ============================================================================
# 5. OPTUNA
# ============================================================================
def _suggest_mesn_params(trial):
    space = MESN_OPTUNA_SPACE
    res_space = space["reservoirs"]
    sensor_names = list(DEFAULT_PARAMS["reservoirs"].keys())

    params = {
        "ridge_alpha":   trial.suggest_float("ridge_alpha", **space["ridge_alpha"]),
        "washout":       trial.suggest_categorical("washout", space["washout"]),
        "state_summary": trial.suggest_categorical("state_summary", space["state_summary"]),
        "min_coverage":  trial.suggest_float("min_coverage", **space["min_coverage"]),
        "random_state":  42,
        "reservoirs":    {},
    }
    for sensor in sensor_names:
        params["reservoirs"][sensor] = {
            "n_reservoir":    trial.suggest_categorical(f"{sensor}_n_reservoir", res_space["n_reservoir"][sensor]),
            "spectral_radius": trial.suggest_float(f"{sensor}_spectral_radius", **res_space["spectral_radius"]),
            "sparsity":        trial.suggest_float(f"{sensor}_sparsity", **res_space["sparsity"]),
            "leak_rate":       trial.suggest_float(f"{sensor}_leak_rate", **res_space["leak_rate"]),
            "input_scaling":   trial.suggest_float(f"{sensor}_input_scaling", **res_space["input_scaling"]),
        }
    return params


def _reconstruct_params(flat_params):
    sensor_names = list(DEFAULT_PARAMS["reservoirs"].keys())
    params = {
        "ridge_alpha":   flat_params["ridge_alpha"],
        "washout":       flat_params["washout"],
        "state_summary": flat_params["state_summary"],
        "min_coverage":  flat_params["min_coverage"],
        "random_state":  42,
        "reservoirs":    {},
    }
    for sensor in sensor_names:
        params["reservoirs"][sensor] = {
            "n_reservoir":     flat_params[f"{sensor}_n_reservoir"],
            "spectral_radius": flat_params[f"{sensor}_spectral_radius"],
            "sparsity":        flat_params[f"{sensor}_sparsity"],
            "leak_rate":       flat_params[f"{sensor}_leak_rate"],
            "input_scaling":   flat_params[f"{sensor}_input_scaling"],
        }
    return params


class _ReadoutProbaWrapper:
    """
    Adapte le Ridge readout de MESN pour exposer predict_proba, requis par
    add_pseudo_labels() de semi_supervised.py (reutilisee telle quelle, non modifiee).
    model.run(x) retourne des scores bruts de regression, pas des probabilites.
    """
    def __init__(self, readout_model):
        self.readout_model = readout_model

    def predict_proba(self, x):
        scores = np.asarray(self.readout_model.run(x), dtype=np.float32)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def optuna_objective(trial, sessions, val_sessions, sensor_specs=None):
    if sensor_specs is None:
        sensor_specs = SENSOR_SPECS
    params = _suggest_mesn_params(trial)
    scores = []

    for val_session in val_sessions:
        model = None
        try:
            train_sessions = [
                s for s in sessions
                if not (s.participant == val_session.participant and s.session == val_session.session)
            ]
            scalers = _make_scalers(train_sessions, sensor_specs=sensor_specs)
            if set(scalers) != set(sensor_specs):
                scores.append(0.0)
                continue

            reservoirs = _build_reservoirs(params, scalers, sensor_specs=sensor_specs)
            x_train, y_train, _, _, _ = extract_mesn_windows(
                train_sessions, reservoirs, scalers, params, sensor_specs=sensor_specs
            )
            x_test, y_test, _, _, _ = extract_mesn_windows(
                [val_session], reservoirs, scalers, params, sensor_specs=sensor_specs
            )

            if len(y_train) > 0:
                x_train_3d = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                x_train_3d, y_train = resample_windows_smote(x_train_3d, y_train)
                x_train = x_train_3d.reshape(x_train_3d.shape[0], x_train_3d.shape[2])

            if len(y_train) == 0 or len(y_test) == 0:
                scores.append(0.0)
                continue

            model = build_readout(params)
            fit_readout(model, x_train, y_train)
            y_pred = predict_readout(model, x_test)
            scores.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

        except Exception as exc:
            print(f"  [WARN] trial split echoue ({type(exc).__name__}: {exc})")
            scores.append(0.0)
        finally:
            _free_memory(model)

    return float(np.mean(scores)) if scores else 0.0


def optimize_hyperparams(sessions, n_trials=MESN_OPTUNA_TRIALS, sensor_specs=None):
    print(f"\nOPTUNA MESN - {n_trials} trials | {MESN_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
    random.seed(42)
    val_sessions = random.sample(sessions, min(MESN_OPTUNA_SESSIONS, len(sessions)))
    print(f"Sessions Optuna: {[(s.participant, s.session) for s in val_sessions]}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, sessions, val_sessions, sensor_specs=sensor_specs),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
        gc_after_trial=True,
        catch=(Exception,),
    )

    print(f"\nBest F1-Macro: {study.best_value:.4f}")
    print(f"Best params  : {study.best_params}")

    OPTUNA_PATH_MESN.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_MESN, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)

    return _reconstruct_params(study.best_params)

# ============================================================================
# 6. VISUALISATION ET SAUVEGARDES
# ============================================================================
def plot_fold(fold_idx, test_part, test_sess, save_dir, f1_mac, bal_acc, y_true, y_pred):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        print(f"  Plot ignore ({type(exc).__name__}: {exc})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"MESN | Fold {fold_idx} - P{test_part} S{test_sess}"
        f" | F1-Macro: {f1_mac:.3f} | Bal.Acc: {bal_acc:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    axes[0].bar(["F1-Macro", "Balanced Acc"], [f1_mac, bal_acc], color=["#2563EB", "#059669"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Scores du fold")
    axes[0].grid(True, axis="y", alpha=0.3)

    labels_present = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[TARGET_NAMES[i] for i in labels_present],
        yticklabels=[TARGET_NAMES[i] for i in labels_present],
        ax=axes[1],
        linewidths=0.5,
        linecolor="gray",
    )
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predit")
    axes[1].set_ylabel("Reel")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plot_path = save_dir / f"fold{fold_idx:02d}_P{test_part}_S{test_sess}_curves.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Courbes sauvegardees: {plot_path}")


def _update_metrics(summary):
    metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
        except json.JSONDecodeError:
            print(f"metrics.json invalide ou vide, reinitialisation: {METRICS_PATH}")

    metrics[MODEL_NAME] = summary
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metriques sauvegardees: {METRICS_PATH}")


# ============================================================================
# 6b. EXPORT ONNX / COMPARAISON EDGE
# ============================================================================
# _compute_classification_metrics, _export_c_array_from_bytes,
# _save_edge_comparison_metrics et _predict_onnx sont dupliquees depuis
# train_esn.py (meme logique exacte) : train_esn.py et MESN.py sont deux
# scripts d'entrainement independants (chacun avec son propre main()), pas
# des modules de bibliotheque partagee -- le projet duplique deja ce genre
# d'utilitaires entre scripts (_free_memory, _ReadoutProbaWrapper existent
# aussi en double), donc un import cross-module casserait cette convention
# sans apporter de reel benefice ici.
def _compute_classification_metrics(y_true, y_pred):
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_fatigue": float(
            precision_score(y_true, y_pred, labels=[3], average="macro", zero_division=0)
        ),
        "recall_fatigue": float(
            recall_score(y_true, y_pred, labels=[3], average="macro", zero_division=0)
        ),
    }


def _save_edge_comparison_metrics(model_name, edge_metrics):
    curr_metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                content = f.read().strip()
            curr_metrics = json.loads(content) if content else {}
        except (json.JSONDecodeError, ValueError):
            print(f"metrics.json invalide ou vide, reinitialisation: {METRICS_PATH}")
            curr_metrics = {}

    curr_metrics.setdefault(model_name, {})
    curr_metrics[model_name]["edge_comparison"] = edge_metrics

    with open(METRICS_PATH, "w") as f:
        json.dump(curr_metrics, f, indent=4)
    print(f"  Comparaison edge sauvegardee: {METRICS_PATH}")


def _export_c_array_from_bytes(file_path, save_dir, model_name):
    with open(file_path, "rb") as f:
        data = f.read()
    c_array = ", ".join(f"0x{b:02x}" for b in data)
    h_path = save_dir / f"{model_name}.h"
    with open(h_path, "w") as f:
        f.write(f"const unsigned char {model_name}_onnx[] = {{{c_array}}};\n")
        f.write(f"const unsigned int {model_name}_onnx_len = {len(data)};\n")
    print(f"  Header C exporte : {h_path}")


def _predict_onnx(onnx_path, X_data):
    """
    Duplique depuis train_esn.py pour la reutilisation demandee. NON utilisee
    par le chemin MESN : le graphe MESN a 6 entrees nommees par capteur (une
    par sensor_name), pas une entree unique comme le graphe ESN mono-capteur
    -- voir _predict_onnx_mesn ci-dessous, qui gere le feed_dict multi-entrees.
    """
    ort_session = ort.InferenceSession(str(onnx_path))
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: X_data.astype(np.float32)})
    return np.asarray(outputs[1], dtype=np.int32)


def _extract_reservoir_weights(reservoir_dict):
    node = reservoir_dict["node"]
    try:
        W_in = node.Win.toarray() if hasattr(node.Win, "toarray") else np.asarray(node.Win)
        W = node.W.toarray() if hasattr(node.W, "toarray") else np.asarray(node.W)
        leak_rate = float(node.lr)
    except AttributeError:
        print(
            f"  [DEBUG] extraction des poids du reservoir '{reservoir_dict.get('sensor', '?')}' "
            f"echouee. Attributs disponibles sur le node: {dir(node)}"
        )
        raise
    return W_in, W, leak_rate


def _extract_readout_weights(readout_model):
    try:
        Wout = readout_model.Wout
        bias = readout_model.bias
    except AttributeError:
        print(
            f"  [DEBUG] extraction des poids du readout echouee. "
            f"Attributs disponibles: {dir(readout_model)}"
        )
        raise
    return Wout, bias


def _nominal_sensor_steps(sensor_specs, window_ms):
    """
    Nombre de pas de temps NOMINAL par capteur pour une entree ONNX de taille
    fixe : round(freq_effective * window_ms / 1000), apres sous-echantillonnage
    par moyennage de blocs. La couverture "attendue" au sens de
    _has_min_coverage reste calculee sur les frequences brutes, AVANT
    sous-echantillonnage. Les fenetres reelles sont tronquees/completees a
    cette taille fixe avant d'entrer dans le graphe (voir
    _extract_raw_sensor_windows).
    """
    return {
        name: max(1, int(round(_effective_freq(name, sensor_specs=sensor_specs) * window_ms / 1000)))
        for name, spec in sensor_specs.items()
    }


def _build_onnx_mesn_graph(reservoirs, scalers, readout_model, params, sensor_specs, model_name):
    """
    Construit un graphe ONNX unique reproduisant le forward MESN complet.

    Pour chaque capteur (dans l'ordre de reservoirs.keys(), qui DOIT correspondre
    exactement a l'ordre de concatenation utilise par np.concatenate(features)
    dans transform_session_windows -- sinon les poids du readout ne correspondent
    plus aux bonnes features) :
      entree brute [valeurs + flags missing] -> separation (Split) -> RobustScaler
      (Sub/Div, valeurs uniquement, flags missing laisses tels quels) -> reservoir
      recurrent deroule sur un nombre de pas FIXE -> resume (last/last_mean/
      last_mean_std, meme logique que _summarize_states).
    Puis concatenation des resumes de tous les capteurs -> Gemm (readout Ridge)
    -> Softmax (probability_tensor) + ArgMax (label_tensor).

    APPROXIMATION IMPORTANTE : chaque capteur a sa propre frequence
    d'echantillonnage (freq dans sensor_specs), donc un nombre de points par
    fenetre different et VARIABLE d'une fenetre reelle a l'autre (couverture
    dependante des trous du signal, cf _has_min_coverage). Le graphe ONNX, lui,
    est STATIQUE et exige une taille d'entree fixe par capteur : on utilise donc
    le nombre de pas NOMINAL (_nominal_sensor_steps), pas le nombre reel de
    points d'une fenetre donnee. A l'inference, les fenetres reelles doivent
    etre tronquees/completees a cette taille fixe (voir _extract_raw_sensor_windows).

    NOTE DOWNSAMPLING : le sous-echantillonnage par moyennage de blocs est fait
    en amont du graphe, dans le chemin Python (_sensor_inputs) et lors de la
    reconstruction des entrees ONNX (_extract_raw_sensor_windows). Le graphe ne
    contient donc pas de noeud AveragePool. Si le firmware STM32 doit consommer
    directement les donnees capteur brutes sans pre-agregation, il faudra
    envisager d'integrer ce moyennage dans le graphe via AveragePool.

    NOTE DE CORRECTION (vs le graphe ESN mono-capteur de train_esn.py, dont le
    pattern de deroulement recurrent a servi de reference) : la reutilisation
    directe de ce pattern se heurte a 3 problemes qui ont ete corriges ici apres
    validation numerique (MAE < 1e-7 vs reservoirpy) :
      1. L'initializer de l'etat initial et la sortie du noeud calcule au premier
         pas de temps portaient le meme nom ("state_0"), ce qui viole la forme
         SSA exigee par onnx.checker.check_model (utilise ici un nom distinct
         "{prefix}_state_init").
      2. Unsqueeze prend "axes" en entree tensor (pas en attribut) a partir de
         l'opset 13 -- l'opset cible ici est 15.
      3. MatMul(prev_state, W) calcule prev_state @ W (convention vecteur-ligne),
         ce qui correspond mathematiquement a W.T @ h et non W @ h : il faut donc
         transposer W (comme W_in) avant de l'utiliser dans MatMul, sans quoi la
         recurrence diverge de reservoirpy des le 2e pas de temps.
    """
    window_ms, _ = _window_ms(None)
    sensor_names = list(reservoirs.keys())
    n_steps_by_sensor = _nominal_sensor_steps(sensor_specs, window_ms)
    state_summary = params.get("state_summary", "last_mean_std")
    washout = int(params.get("washout", 0))
    num_classes = len(LABEL_MAPPING)

    Wout, bias = _extract_readout_weights(readout_model)
    # Wout a deja la forme (n_features_in, n_classes) -- convention verifiee
    # numeriquement (model.run(x) == Gemm(x, Wout, bias) avec transB=0 par
    # defaut, MAE < 1e-7). PAS de transposition ici, contrairement au pattern
    # de train_esn.py qui transpose Wout avant Gemm : applique tel quel, ce
    # dernier produit un logits de shape (batch, n_features_in) au lieu de
    # (batch, n_classes) (confirme empiriquement : erreur "Invalid bias shape
    # for broadcast" / mismatch de shape a l'execution).
    Wout = np.asarray(Wout, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32).reshape(-1)

    nodes = []
    initializers = [
        onnx.numpy_helper.from_array(Wout, "Wout"),
        onnx.numpy_helper.from_array(bias, "bias"),
    ]
    graph_inputs = []
    sensor_summaries = []

    for sensor_name in sensor_names:
        prefix = sensor_name
        reservoir = reservoirs[sensor_name]
        W_in, W, leak_rate = _extract_reservoir_weights(reservoir)
        W_in_T = np.asarray(np.asarray(W_in, dtype=np.float32).T, dtype=np.float32)
        W_T = np.asarray(np.asarray(W, dtype=np.float32).T, dtype=np.float32)
        n_reservoir = W_T.shape[0]

        value_cols = list(sensor_specs[sensor_name]["cols"].values())
        n_features = len(value_cols)
        n_steps = n_steps_by_sensor[sensor_name]

        scaler = scalers[sensor_name]
        center = np.asarray(scaler.center_, dtype=np.float32).reshape(1, 1, n_features)
        scale = np.asarray(scaler.scale_, dtype=np.float32).reshape(1, 1, n_features)

        input_name = f"{prefix}_input"
        graph_inputs.append(
            onnx.helper.make_tensor_value_info(
                input_name, onnx.TensorProto.FLOAT, [None, n_steps, n_features * 2]
            )
        )

        initializers.extend(
            [
                onnx.numpy_helper.from_array(
                    np.asarray([n_features, n_features], dtype=np.int64), f"{prefix}_split_sizes"
                ),
                onnx.numpy_helper.from_array(center, f"{prefix}_center"),
                onnx.numpy_helper.from_array(scale, f"{prefix}_scale"),
                onnx.numpy_helper.from_array(W_in_T, f"{prefix}_W_in_T"),
                onnx.numpy_helper.from_array(W_T, f"{prefix}_W_T"),
                onnx.numpy_helper.from_array(
                    np.asarray([1.0 - float(leak_rate)], dtype=np.float32), f"{prefix}_one_minus_leak"
                ),
                onnx.numpy_helper.from_array(np.asarray([float(leak_rate)], dtype=np.float32), f"{prefix}_leak"),
                onnx.numpy_helper.from_array(np.zeros((1, n_reservoir), dtype=np.float32), f"{prefix}_state_init"),
                onnx.numpy_helper.from_array(np.asarray([0], dtype=np.int64), f"{prefix}_axis0"),
            ]
        )

        # Separation valeurs / flags missing, normalisation RobustScaler sur les
        # valeurs uniquement (flags missing laisses 0/1 tels quels, cf _sensor_inputs).
        nodes.extend(
            [
                onnx.helper.make_node(
                    "Split",
                    [input_name, f"{prefix}_split_sizes"],
                    [f"{prefix}_values_raw", f"{prefix}_missing"],
                    axis=2,
                    name=f"{prefix}_split",
                ),
                onnx.helper.make_node("Sub", [f"{prefix}_values_raw", f"{prefix}_center"], [f"{prefix}_centered"]),
                onnx.helper.make_node("Div", [f"{prefix}_centered", f"{prefix}_scale"], [f"{prefix}_scaled"]),
                onnx.helper.make_node(
                    "Concat", [f"{prefix}_scaled", f"{prefix}_missing"], [f"{prefix}_scaled_input"], axis=2
                ),
            ]
        )

        # Recurrence du reservoir deroulee sur n_steps pas fixes :
        # h(t) = (1-leak)*h(t-1) + leak*tanh(W_in @ x(t) + W @ h(t-1))
        # (W_in et W transposes pour la convention MatMul vecteur-ligne, cf
        # note de correction ci-dessus)
        prev_state = f"{prefix}_state_init"
        kept_states = []
        for t in range(n_steps):
            initializers.append(
                onnx.numpy_helper.from_array(np.asarray(t, dtype=np.int64), f"{prefix}_time_idx_{t}")
            )
            nodes.extend(
                [
                    onnx.helper.make_node(
                        "Gather", [f"{prefix}_scaled_input", f"{prefix}_time_idx_{t}"], [f"{prefix}_x_t_{t}"], axis=1
                    ),
                    onnx.helper.make_node(
                        "MatMul", [f"{prefix}_x_t_{t}", f"{prefix}_W_in_T"], [f"{prefix}_in_proj_{t}"]
                    ),
                    onnx.helper.make_node("MatMul", [prev_state, f"{prefix}_W_T"], [f"{prefix}_rec_proj_{t}"]),
                    onnx.helper.make_node(
                        "Add", [f"{prefix}_in_proj_{t}", f"{prefix}_rec_proj_{t}"], [f"{prefix}_preact_{t}"]
                    ),
                    onnx.helper.make_node("Tanh", [f"{prefix}_preact_{t}"], [f"{prefix}_candidate_{t}"]),
                    onnx.helper.make_node(
                        "Mul", [prev_state, f"{prefix}_one_minus_leak"], [f"{prefix}_prev_scaled_{t}"]
                    ),
                    onnx.helper.make_node(
                        "Mul", [f"{prefix}_candidate_{t}", f"{prefix}_leak"], [f"{prefix}_candidate_scaled_{t}"]
                    ),
                    onnx.helper.make_node(
                        "Add", [f"{prefix}_prev_scaled_{t}", f"{prefix}_candidate_scaled_{t}"], [f"{prefix}_state_{t}"]
                    ),
                ]
            )
            prev_state = f"{prefix}_state_{t}"
            if t >= washout:
                kept_states.append(prev_state)

        if not kept_states:
            kept_states = [prev_state]

        last_state = kept_states[-1]
        summary_inputs = [last_state]

        if state_summary in {"last_mean", "last_mean_std"}:
            unsqueezed = []
            for idx, state_name in enumerate(kept_states):
                out_name = f"{prefix}_seq_{idx}"
                nodes.append(onnx.helper.make_node("Unsqueeze", [state_name, f"{prefix}_axis0"], [out_name]))
                unsqueezed.append(out_name)
            nodes.append(onnx.helper.make_node("Concat", unsqueezed, [f"{prefix}_states_seq"], axis=0))
            nodes.append(
                onnx.helper.make_node(
                    "ReduceMean", [f"{prefix}_states_seq"], [f"{prefix}_states_mean"], axes=[0], keepdims=0
                )
            )
            summary_inputs.append(f"{prefix}_states_mean")

        if state_summary == "last_mean_std":
            nodes.extend(
                [
                    onnx.helper.make_node(
                        "Sub", [f"{prefix}_states_seq", f"{prefix}_states_mean"], [f"{prefix}_centered_states"]
                    ),
                    onnx.helper.make_node(
                        "Mul", [f"{prefix}_centered_states", f"{prefix}_centered_states"], [f"{prefix}_var_terms"]
                    ),
                    onnx.helper.make_node(
                        "ReduceMean", [f"{prefix}_var_terms"], [f"{prefix}_states_var"], axes=[0], keepdims=0
                    ),
                    onnx.helper.make_node("Sqrt", [f"{prefix}_states_var"], [f"{prefix}_states_std"]),
                ]
            )
            summary_inputs.append(f"{prefix}_states_std")
        elif state_summary not in {"last", "last_mean"}:
            raise ValueError(f"state_summary inconnu: {state_summary}")

        if len(summary_inputs) == 1:
            nodes.append(onnx.helper.make_node("Identity", [summary_inputs[0]], [f"{prefix}_summary"]))
        else:
            nodes.append(onnx.helper.make_node("Concat", summary_inputs, [f"{prefix}_summary"], axis=1))

        sensor_summaries.append(f"{prefix}_summary")

    # Concatenation des resumes par capteur, dans l'ordre de reservoirs.keys().
    nodes.append(onnx.helper.make_node("Concat", sensor_summaries, ["summary_all"], axis=1))
    nodes.extend(
        [
            onnx.helper.make_node("Gemm", ["summary_all", "Wout", "bias"], ["logits"], alpha=1.0, beta=1.0),
            onnx.helper.make_node("Softmax", ["logits"], ["probability_tensor"], axis=1),
            onnx.helper.make_node("ArgMax", ["logits"], ["label_tensor"], axis=1, keepdims=0),
        ]
    )

    graph = onnx.helper.make_graph(
        nodes,
        model_name,
        graph_inputs,
        [
            onnx.helper.make_tensor_value_info("probability_tensor", onnx.TensorProto.FLOAT, [None, num_classes]),
            onnx.helper.make_tensor_value_info("label_tensor", onnx.TensorProto.INT64, [None]),
        ],
        initializers,
    )
    model = onnx.helper.make_model(
        graph,
        producer_name=f"{MODEL_NAME}_manual_export",
        opset_imports=[onnx.helper.make_operatorsetid("", 15)],
    )
    onnx.checker.check_model(model)
    return model


def _extract_raw_sensor_windows(sessions, params, sensor_specs, n_steps_by_sensor):
    """
    Reconstruit, pour chaque fenetre valide (meme critere que
    transform_session_windows : label connu et != "unlabeled", couverture
    minimale par capteur via _has_min_coverage), les tenseurs BRUTS (valeurs +
    flags missing, NON normalises -- la normalisation RobustScaler est faite
    par le graphe ONNX lui-meme) par capteur, tronques/completes a la taille
    fixe n_steps_by_sensor utilisee par le graphe (_nominal_sensor_steps).

    Necessaire car extract_mesn_windows ne retourne que les features DEJA
    resumees par les reservoirs, alors que le graphe ONNX prend en entree les
    fenetres brutes par capteur -- sans cette reconstruction, impossible de
    generer des predictions ONNX comparables au chemin Python.

    Les fenetres reelles plus longues que le nominal sont tronquees (on garde
    les pas les plus recents) ; les plus courtes sont completees en tete par
    des zeros avec le flag missing force a 1 (pas synthetiques marques
    "manquants", coherent avec la semantique de ces flags ailleurs dans le code).
    """
    window_ms, step_ms = _window_ms(None)
    sensor_names = list(sensor_specs.keys())

    windows_by_sensor = {name: [] for name in sensor_names}
    labels = []
    meta = []

    for session in sessions:
        start_bound, end_bound = _session_time_bounds(session)
        if start_bound is None or end_bound is None or end_bound - start_bound < window_ms:
            continue

        start = float(start_bound)
        while start + window_ms <= end_bound:
            end = start + window_ms
            label = _label_at(session.intervals, start, end)
            if label is None or label == "unlabeled":
                start += step_ms
                continue

            per_sensor = {}
            valid = True
            for sensor_name in sensor_names:
                df_sensor = session.sensors.get(sensor_name)
                value_cols = list(sensor_specs[sensor_name]["cols"].values())
                missing_cols = [f"{col}_missing" for col in value_cols]
                required_cols = value_cols + missing_cols
                if df_sensor is None or not all(col in df_sensor.columns for col in required_cols):
                    valid = False
                    break

                x_values = _sensor_window(df_sensor, start, end, value_cols)
                if not _has_min_coverage(sensor_name, x_values, window_ms, params, sensor_specs=sensor_specs):
                    valid = False
                    break

                x_missing = _sensor_window(df_sensor, start, end, missing_cols)
                factor = _downsample_factor(sensor_name, sensor_specs=sensor_specs)
                x_values = _downsample_block_average(x_values, factor)
                x_missing = _downsample_block_average(x_missing, factor)
                x_raw = np.concatenate([x_values, x_missing.astype(np.float32)], axis=1).astype(np.float32)

                n_steps = n_steps_by_sensor[sensor_name]
                n_features = len(value_cols)
                x_fixed = _fix_sensor_window_length(x_raw, n_steps, n_features)

                per_sensor[sensor_name] = x_fixed

            if valid:
                for sensor_name in sensor_names:
                    windows_by_sensor[sensor_name].append(per_sensor[sensor_name])
                labels.append(LABEL_MAPPING[label])
                meta.append(
                    {
                        "participant": session.participant,
                        "session": session.session,
                        "start_ms": start,
                        "end_ms": end,
                        "label": label,
                    }
                )

            start += step_ms

    for sensor_name in sensor_names:
        n_steps = n_steps_by_sensor[sensor_name]
        n_features = len(sensor_specs[sensor_name]["cols"])
        windows_by_sensor[sensor_name] = (
            np.stack(windows_by_sensor[sensor_name]).astype(np.float32)
            if windows_by_sensor[sensor_name]
            else np.empty((0, n_steps, n_features * 2), dtype=np.float32)
        )

    return windows_by_sensor, np.asarray(labels, dtype=np.int32), meta


def _predict_onnx_mesn(onnx_path, windows_by_sensor):
    """
    Variante multi-entrees de _predict_onnx (voir sa docstring) : construit un
    feed_dict {"{sensor}_input": tenseur} au lieu d'un tenseur unique, pour
    correspondre aux 6 entrees nommees du graphe MESN.
    """
    ort_session = ort.InferenceSession(str(onnx_path))
    feed = {f"{name}_input": arr.astype(np.float32) for name, arr in windows_by_sensor.items()}
    outputs = ort_session.run(None, feed)
    return np.asarray(outputs[1], dtype=np.int32)


def _validate_onnx_mesn(onnx_model, reservoirs, scalers, readout_model, params, sensor_specs, sessions_sample):
    """
    Validation numerique OBLIGATOIRE du graphe ONNX MESN contre le chemin
    Python (reservoirs reservoirpy + readout Ridge). Le graphe ONNX ici est
    nettement plus complexe que celui de train_esn.py (6 reservoirs a
    frequences differentes, approximation du nombre de pas fixe par capteur) :
    le risque d'erreur de construction est eleve, donc chaque fenetre est
    comparee individuellement (session.run par fenetre), pas en lot.
    """
    print("\n  --- Validation ONNX MESN ---")
    x_python, y_python, _, _, _ = extract_mesn_windows(
        sessions_sample, reservoirs, scalers, params, sensor_specs=sensor_specs
    )
    if len(y_python) == 0:
        print("  Validation ONNX MESN : aucune fenetre labellisee dans l'echantillon, validation ignoree.")
        return

    window_ms, _ = _window_ms(None)
    n_steps_by_sensor = _nominal_sensor_steps(sensor_specs, window_ms)
    windows_by_sensor, y_raw, _ = _extract_raw_sensor_windows(sessions_sample, params, sensor_specs, n_steps_by_sensor)

    n_compare = min(len(y_python), len(y_raw))
    if n_compare == 0:
        print("  Validation ONNX MESN : impossible de comparer (0 fenetre commune).")
        return
    if len(y_python) != len(y_raw) or not np.array_equal(y_python[:n_compare], y_raw[:n_compare]):
        print(
            f"  WARNING Validation ONNX MESN : desalignement potentiel entre les fenetres "
            f"Python ({len(y_python)}) et ONNX ({len(y_raw)}) -- comparaison limitee aux "
            f"{n_compare} premieres fenetres communes, resultat a interpreter avec prudence."
        )

    proba_wrapper = _ReadoutProbaWrapper(readout_model)
    python_proba = proba_wrapper.predict_proba(x_python[:n_compare])
    y_pred_python = np.argmax(python_proba, axis=1).astype(np.int32)

    ort_session = ort.InferenceSession(onnx_model.SerializeToString())
    y_pred_onnx = np.zeros(n_compare, dtype=np.int32)
    onnx_proba = np.zeros((n_compare, python_proba.shape[1]), dtype=np.float32)
    for i in range(n_compare):
        feed = {
            f"{name}_input": windows_by_sensor[name][i : i + 1].astype(np.float32) for name in windows_by_sensor
        }
        outputs = ort_session.run(None, feed)
        onnx_proba[i] = outputs[0][0]
        y_pred_onnx[i] = int(outputs[1][0])

    agreement = float(np.mean(y_pred_python == y_pred_onnx)) * 100
    mae = float(np.mean(np.abs(python_proba - onnx_proba)))

    print(f"  Validation ONNX MESN : {agreement:.1f}% d'accord sur {n_compare} fenetres, MAE={mae:.6f}")
    if agreement < 95.0:
        print(
            f"  WARNING : accord ONNX/Python ({agreement:.1f}%) sous le seuil de 95%. "
            f"Verifier la construction du graphe (approximation du nombre de pas fixe "
            f"par capteur, ordre de concatenation, extraction des poids)."
        )


def train_global_model(sessions, params):
    print("\n" + "=" * 60 + f"\nMODELE GLOBAL - {MODEL_NAME}\n" + "=" * 60)

    try:
        scalers = _make_scalers(sessions, sensor_specs=SENSOR_SPECS_CSV)
        if set(scalers) != set(SENSOR_SPECS_CSV):
            missing = sorted(set(SENSOR_SPECS_CSV) - set(scalers))
            print(f"  Modele global echoue : scalers manquants pour {missing}")
            return

        reservoirs = _build_reservoirs(params, scalers, sensor_specs=SENSOR_SPECS_CSV)

        idx = np.random.default_rng(42).permutation(len(sessions))
        split = int(0.9 * len(sessions))
        train_sessions = [sessions[i] for i in idx[:split]]
        val_sessions = [sessions[i] for i in idx[split:]]

        x_train, y_train, _, x_unlabeled, _ = extract_mesn_windows(
            train_sessions, reservoirs, scalers, params, sensor_specs=SENSOR_SPECS_CSV
        )
        x_val, y_val, _, _, _ = extract_mesn_windows(
            val_sessions, reservoirs, scalers, params, sensor_specs=SENSOR_SPECS_CSV
        )

        if len(y_train) == 0 or len(y_val) == 0:
            print("  Modele global echoue : train/val vide")
            return

        if len(y_train) > 0:
            x_train_3d = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
            x_train_3d, y_train = resample_windows_smote(x_train_3d, y_train)
            x_train = x_train_3d.reshape(x_train_3d.shape[0], x_train_3d.shape[2])

        model = build_readout(params)
        fit_readout(model, x_train, y_train)

        if len(x_unlabeled) > 0 and x_train.shape[1] == x_unlabeled.shape[1]:
            proba_wrapper = _ReadoutProbaWrapper(model)
            x_train_v2, y_train_v2, pseudo_y, _ = add_pseudo_labels(proba_wrapper, x_train, y_train, x_unlabeled)
            if len(pseudo_y) > 0:
                _free_memory(model)
                model = build_readout(params)
                fit_readout(model, x_train_v2, y_train_v2)

        models_dir = MODELS_DIR / "MESN"
        models_dir.mkdir(parents=True, exist_ok=True)
        edge_metrics = {}

        # Backup natif
        pkl_path = None
        try:
            pkl_path = models_dir / f"{MODEL_NAME}_global.pkl"
            joblib.dump({"reservoirs": reservoirs, "readout": model, "scalers": scalers}, pkl_path)
            print(f"  Pickle backup : {pkl_path}")
        except Exception as exc:
            print(f"  Pickle backup echoue ({type(exc).__name__}: {exc})")

        y_pred_native = predict_readout(model, x_val)
        edge_metrics["native_mesn"] = _compute_classification_metrics(y_val, y_pred_native)
        if pkl_path is not None and pkl_path.exists():
            edge_metrics["native_mesn"]["model_size_kb"] = round(pkl_path.stat().st_size / 1024, 2)

        # Export ONNX
        try:
            onnx_model = _build_onnx_mesn_graph(
                reservoirs, scalers, model, params, SENSOR_SPECS_CSV, f"{MODEL_NAME}_global"
            )
            _validate_onnx_mesn(onnx_model, reservoirs, scalers, model, params, SENSOR_SPECS_CSV, val_sessions[:10])

            onnx_path = models_dir / f"{MODEL_NAME}_global.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"  ONNX exporte : {onnx_path}")

            _export_c_array_from_bytes(onnx_path, models_dir, f"{MODEL_NAME}_global")

            # NOTE / ECART VOLONTAIRE : contrairement a train_esn.py (une seule
            # entree, x_val directement utilisable), le graphe MESN a 6 entrees
            # nommees par capteur qui attendent des fenetres BRUTES, alors que
            # x_val est deja resume par les reservoirs (sortie de
            # extract_mesn_windows). _predict_onnx(onnx_path, x_val) ne peut donc
            # pas fonctionner tel quel ici : on reconstruit les fenetres brutes
            # par capteur (meme logique que _validate_onnx_mesn) et on utilise
            # _predict_onnx_mesn (multi-entrees) a la place.
            window_ms, _ = _window_ms(None)
            n_steps_by_sensor = _nominal_sensor_steps(SENSOR_SPECS_CSV, window_ms)
            windows_by_sensor_val, y_val_onnx, _ = _extract_raw_sensor_windows(
                val_sessions, params, SENSOR_SPECS_CSV, n_steps_by_sensor
            )
            if len(y_val_onnx) != len(y_val) or not np.array_equal(y_val_onnx, y_val):
                print(
                    f"  WARNING : desalignement fenetres pour edge_metrics['onnx'] "
                    f"({len(y_val)} Python vs {len(y_val_onnx)} ONNX) -- metriques ONNX "
                    f"calculees sur les fenetres reconstruites (y_val_onnx), a interpreter avec prudence."
                )

            y_pred_onnx = _predict_onnx_mesn(onnx_path, windows_by_sensor_val)
            edge_metrics["onnx"] = _compute_classification_metrics(y_val_onnx, y_pred_onnx)
            edge_metrics["onnx"]["model_size_kb"] = round(onnx_path.stat().st_size / 1024, 2)

        except Exception as exc:
            print(f"  Export ONNX echoue ({type(exc).__name__}: {exc})")
            traceback.print_exc()

        _save_edge_comparison_metrics(MODEL_NAME, edge_metrics)
        print(f"  Modele global MESN traite (native + export ONNX si reussi).")

    except Exception as exc:
        print(f"  Modele global echoue ({type(exc).__name__}: {exc})")
        traceback.print_exc()


# ============================================================================
# 7. BOUCLE LOSO
# ============================================================================
def main():
    print("\n" + "=" * 60 + f"\nTRAIN LOSO - {MODEL_NAME}\n" + "=" * 60)
    if not DATA_RAW.exists():
        return print(f"Raw data non trouve: {DATA_RAW}")

    sessions = load_sessions_from_csv(OUTPUT_PATH_NO_SMOTE_FREQ_NO_SYNC, sensor_specs=SENSOR_SPECS_CSV)
    if not sessions:
        return print("Aucune session raw disponible.")

    params = optimize_hyperparams(sessions, sensor_specs=SENSOR_SPECS_CSV) if USE_OPTUNA_MESN else DEFAULT_PARAMS

    unique_sessions = [(s.participant, s.session) for s in sessions]
    print(f"Sessions chargees: {len(unique_sessions)}")

    all_metrics = []
    plots_dir = BASE_DIR / "training_curves" / MODEL_NAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (test_part, test_sess) in enumerate(unique_sessions, start=1):
        print(f"\n--- FOLD {fold_idx}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")
        train_sessions = [
            s for s in sessions if not (s.participant == test_part and s.session == test_sess)
        ]
        test_sessions = [
            s for s in sessions if s.participant == test_part and s.session == test_sess
        ]

        model = None
        try:
            scalers = _make_scalers(train_sessions, sensor_specs=SENSOR_SPECS_CSV)
            if set(scalers) != set(SENSOR_SPECS_CSV):
                missing = sorted(set(SENSOR_SPECS_CSV) - set(scalers))
                print(f"  WARNING: scalers manquants pour {missing}. Fold ignore.")
                continue

            reservoirs = _build_reservoirs(params, scalers, sensor_specs=SENSOR_SPECS_CSV)
            x_train, y_train, _, x_unlabeled, _ = extract_mesn_windows(
                train_sessions, reservoirs, scalers, params, sensor_specs=SENSOR_SPECS_CSV
            )
            x_test, y_test, _, _, _ = extract_mesn_windows(
                test_sessions, reservoirs, scalers, params, sensor_specs=SENSOR_SPECS_CSV
            )

            if len(y_train) > 0:
                x_train_3d = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                x_train_3d, y_train = resample_windows_smote(x_train_3d, y_train)
                x_train = x_train_3d.reshape(x_train_3d.shape[0], x_train_3d.shape[2])

            if len(y_train) == 0 or len(y_test) == 0:
                print("  WARNING: train/test vide. Fold ignore.")
                continue

            print(f"  Train: {len(x_train)} fenetres | Test: {len(x_test)} fenetres")
            model = build_readout(params)
            fit_readout(model, x_train, y_train)

            if len(x_unlabeled) > 0 and x_train.shape[1] == x_unlabeled.shape[1]:
                proba_wrapper = _ReadoutProbaWrapper(model)
                x_train_v2, y_train_v2, pseudo_y, _ = add_pseudo_labels(proba_wrapper, x_train, y_train, x_unlabeled)
                if len(pseudo_y) > 0:
                    _free_memory(model)
                    model = build_readout(params)
                    fit_readout(model, x_train_v2, y_train_v2)

            y_pred = predict_readout(model, x_test)

            f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"  F1-Macro: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}")

            labels_present = np.unique(np.concatenate([y_test, y_pred]))
            print(
                "\n"
                + classification_report(
                    y_test,
                    y_pred,
                    labels=labels_present,
                    target_names=[TARGET_NAMES[i] for i in labels_present],
                    zero_division=0,
                )
            )

            plot_fold(fold_idx, test_part, test_sess, plots_dir, f1_mac, bal_acc, y_test, y_pred)
            all_metrics.append(
                {
                    "Fold": fold_idx,
                    "Participant": int(test_part),
                    "Session": int(test_sess),
                    "F1_Macro": float(f1_mac),
                    "Balanced_Accuracy": float(bal_acc),
                    "Train_Windows": int(len(x_train)),
                    "Test_Windows": int(len(x_test)),
                }
            )

        except Exception as exc:
            print(f"  Fold {fold_idx} echoue ({type(exc).__name__}: {exc})")
        finally:
            _free_memory(model)

    if not all_metrics:
        return print("Aucun fold complete.")

    df_res = pd.DataFrame(all_metrics)
    print("\n" + "=" * 60 + f"\nRESULTATS FINAUX - {MODEL_NAME}\n" + "=" * 60)
    print(df_res[["F1_Macro", "Balanced_Accuracy"]].describe().loc[["mean", "std"]])

    summary = {
        "mean_f1_macro": float(df_res["F1_Macro"].mean()),
        "std_f1_macro": float(df_res["F1_Macro"].std()),
        "mean_balanced_accuracy": float(df_res["Balanced_Accuracy"].mean()),
        "std_balanced_accuracy": float(df_res["Balanced_Accuracy"].std()),
        "folds": all_metrics,
        "params": params,
        "sensor_specs": {
            name: {
                "file": spec["file"],
                "cols": spec["cols"],
                "freq": spec["freq"],
            }
            for name, spec in SENSOR_SPECS.items()
        },
    }
    _update_metrics(summary)
    train_global_model(sessions, params)


if __name__ == "__main__":
    main()
