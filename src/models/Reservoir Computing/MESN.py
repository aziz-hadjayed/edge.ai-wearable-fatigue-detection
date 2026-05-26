import gc
import json
import random
import sys
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

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 1. CONSTANTES
# ============================================================================
MODEL_NAME = "MESN"
LABEL_MAPPING = {"baseline": 0, "activity": 1, "fatigue": 2}
TARGET_NAMES = ["baseline", "activity", "fatigue"]

DEFAULT_PARAMS = {
    "ridge_alpha": 5.0,
    "washout": 5,
    "state_summary": "last_mean_std",
    "min_coverage": 0.35,
    "random_state": 42,
    "reservoirs": {
        "acc": {
            "n_reservoir": 80,
            "spectral_radius": 0.75,
            "sparsity": 0.90,
            "leak_rate": 0.35,
            "input_scaling": 0.50,
        },
        "eda": {
            "n_reservoir": 50,
            "spectral_radius": 0.70,
            "sparsity": 0.90,
            "leak_rate": 0.20,
            "input_scaling": 0.60,
        },
        "hr": {
            "n_reservoir": 40,
            "spectral_radius": 0.65,
            "sparsity": 0.90,
            "leak_rate": 0.15,
            "input_scaling": 0.50,
        },
        "ibi": {
            "n_reservoir": 40,
            "spectral_radius": 0.60,
            "sparsity": 0.90,
            "leak_rate": 0.20,
            "input_scaling": 0.50,
        },
        "temp": {
            "n_reservoir": 40,
            "spectral_radius": 0.65,
            "sparsity": 0.90,
            "leak_rate": 0.10,
            "input_scaling": 0.45,
        },
        "breathing": {
            "n_reservoir": 70,
            "spectral_radius": 0.80,
            "sparsity": 0.92,
            "leak_rate": 0.30,
            "input_scaling": 0.50,
        },
    },
}

SENSOR_SPECS = {
    "acc": {
        "file": "wrist_acc.csv",
        "cols": {"ax": "acc_x", "ay": "acc_y", "az": "acc_z"},
        "freq": 32.0,
        "log1p": [],
    },
    "eda": {
        "file": "wrist_eda.csv",
        "cols": {"eda": "eda"},
        "freq": 4.0,
        "log1p": ["eda"],
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
        "freq": 0.59,
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


def _make_scalers(train_sessions):
    scalers = {}
    for sensor_name, spec in SENSOR_SPECS.items():
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


def _build_reservoirs(params, scalers):
    reservoirs = {}
    for sensor_name, scaler in scalers.items():
        reservoirs[sensor_name] = _build_reservoir(
            sensor_name,
            params,
            input_dim=len(SENSOR_SPECS[sensor_name]["cols"]) * 2,
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


def _sensor_inputs(df, sensor_name, start_ms, end_ms, scalers):
    value_cols = list(SENSOR_SPECS[sensor_name]["cols"].values())
    missing_cols = [f"{col}_missing" for col in value_cols]
    x_values = _sensor_window(df, start_ms, end_ms, value_cols)
    x_missing = _sensor_window(df, start_ms, end_ms, missing_cols)
    x_values = scalers[sensor_name].transform(x_values).astype(np.float32)
    return np.concatenate([x_values, x_missing.astype(np.float32)], axis=1)


def _has_min_coverage(sensor_name, x, window_ms, params):
    expected = SENSOR_SPECS[sensor_name]["freq"] * (window_ms / 1000)
    min_samples = max(2, int(round(expected * float(params.get("min_coverage", 0.35)))))
    if SENSOR_SPECS[sensor_name]["freq"] < 1:
        min_samples = max(2, min(min_samples, 8))
    return len(x) >= min_samples


def transform_session_windows(session, reservoirs, scalers, params):
    window_ms, step_ms = _window_ms(session.participant)
    start_bound, end_bound = _session_time_bounds(session)
    if start_bound is None or end_bound is None or end_bound - start_bound < window_ms:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int32), []

    x_all, y_all, meta = [], [], []
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
            value_cols = list(SENSOR_SPECS[sensor_name]["cols"].values())
            missing_cols = [f"{col}_missing" for col in value_cols]
            required_cols = value_cols + missing_cols
            if df_sensor is None or not all(col in df_sensor.columns for col in required_cols):
                valid = False
                break

            x_sensor = _sensor_window(df_sensor, start, end, value_cols)
            if not _has_min_coverage(sensor_name, x_sensor, window_ms, params):
                valid = False
                break

            x_input = _sensor_inputs(df_sensor, sensor_name, start, end, scalers)
            states = _reservoir_states(x_input, reservoirs[sensor_name], washout)
            features.append(_summarize_states(states, summary))

        if valid:
            x_all.append(np.concatenate(features).astype(np.float32))
            y_all.append(LABEL_MAPPING[label])
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

    if not x_all:
        feature_dim = sum(_feature_size(params, name) for name in sensor_names)
        return np.empty((0, feature_dim), dtype=np.float32), np.array([], dtype=np.int32), []
    return np.asarray(x_all, dtype=np.float32), np.asarray(y_all, dtype=np.int32), meta


def extract_mesn_windows(sessions, reservoirs, scalers, params):
    xs, ys, metas = [], [], []
    for session in sessions:
        x_session, y_session, meta_session = transform_session_windows(
            session, reservoirs, scalers, params
        )
        if len(y_session) > 0:
            xs.append(x_session)
            ys.append(y_session)
            metas.extend(meta_session)

    if not xs:
        feature_dim = sum(_feature_size(params, name) for name in reservoirs.keys())
        return np.empty((0, feature_dim), dtype=np.float32), np.array([], dtype=np.int32), []
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.int32), metas


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


def _to_onehot(y, num_classes=3):
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


def optuna_objective(trial, sessions, val_sessions):
    params = _suggest_mesn_params(trial)
    scores = []

    for val_session in val_sessions:
        model = None
        try:
            train_sessions = [
                s for s in sessions
                if not (s.participant == val_session.participant and s.session == val_session.session)
            ]
            scalers = _make_scalers(train_sessions)
            if set(scalers) != set(SENSOR_SPECS):
                scores.append(0.0)
                continue

            reservoirs = _build_reservoirs(params, scalers)
            x_train, y_train, _ = extract_mesn_windows(train_sessions, reservoirs, scalers, params)
            x_test, y_test, _ = extract_mesn_windows([val_session], reservoirs, scalers, params)

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


def optimize_hyperparams(sessions, n_trials=MESN_OPTUNA_TRIALS):
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
        lambda trial: optuna_objective(trial, sessions, val_sessions),
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


def train_global_model(sessions, params):
    print("\n" + "=" * 60 + f"\nMODELE GLOBAL - {MODEL_NAME}\n" + "=" * 60)
    scalers = _make_scalers(sessions)
    reservoirs = _build_reservoirs(params, scalers)
    x_all, y_all, _ = extract_mesn_windows(sessions, reservoirs, scalers, params)
    if len(y_all) == 0:
        print("  Aucun exemple global disponible.")
        return

    model = build_readout(params)
    fit_readout(model, x_all, y_all)

    models_dir = MODELS_DIR / MODEL_NAME
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{MODEL_NAME}_global.pkl"
    joblib.dump(
        {
            "model_name": MODEL_NAME,
            "readout": model,
            "reservoirs": reservoirs,
            "scalers": scalers,
            "sensor_specs": SENSOR_SPECS,
            "label_mapping": LABEL_MAPPING,
            "target_names": TARGET_NAMES,
            "params": params,
        },
        model_path,
    )
    print(f"  Modele global sauvegarde: {model_path}")


# ============================================================================
# 7. BOUCLE LOSO
# ============================================================================
def main():
    print("\n" + "=" * 60 + f"\nTRAIN LOSO - {MODEL_NAME}\n" + "=" * 60)
    if not DATA_RAW.exists():
        return print(f"Raw data non trouve: {DATA_RAW}")

    sessions = load_raw_sessions(DATA_RAW)
    if not sessions:
        return print("Aucune session raw disponible.")

    params = optimize_hyperparams(sessions) if USE_OPTUNA_MESN else DEFAULT_PARAMS

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
            scalers = _make_scalers(train_sessions)
            if set(scalers) != set(SENSOR_SPECS):
                missing = sorted(set(SENSOR_SPECS) - set(scalers))
                print(f"  WARNING: scalers manquants pour {missing}. Fold ignore.")
                continue

            reservoirs = _build_reservoirs(params, scalers)
            x_train, y_train, _ = extract_mesn_windows(train_sessions, reservoirs, scalers, params)
            x_test, y_test, _ = extract_mesn_windows(test_sessions, reservoirs, scalers, params)

            if len(y_train) == 0 or len(y_test) == 0:
                print("  WARNING: train/test vide. Fold ignore.")
                continue

            print(f"  Train: {len(x_train)} fenetres | Test: {len(x_test)} fenetres")
            model = build_readout(params)
            fit_readout(model, x_train, y_train)
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
