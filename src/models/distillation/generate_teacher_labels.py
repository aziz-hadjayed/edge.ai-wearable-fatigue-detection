"""
Étape 1 de la distillation de connaissance : génère les soft labels de
l'ensemble teacher (LGBM + MESN + CNN_LSTM, tous déjà entraînés — AUCUN
réentraînement ici) pour chaque fenêtre student (CNN_1D).

Problème central : les 3 teachers et le student n'opèrent pas sur la même
définition de fenêtre (cf. docstring de _extract_student_windows et
_mesn_predict_interval). On prend donc les fenêtres CNN_1D comme référence
(participant, session, start_ms, end_ms) et on recalcule l'input propre à
chaque teacher pour ce même intervalle temporel. Une fenêtre est écartée
(jamais imputée) dès qu'un des 3 teachers ne peut pas produire de prédiction
valide sur cet intervalle (session manquante côté MESN, couverture capteur
insuffisante, cf. _has_min_coverage).

Sortie : data/03_processed/teacher_soft_labels.npz
"""
import json
import os
import sys
import time
from pathlib import Path

# CNN_LSTM_global_float32.keras a été sauvegardé par train_cnn1_lstm.py avec
# TF_USE_LEGACY_KERAS=1 (tf_keras) — même variable requise ici AVANT le import
# tensorflow pour pouvoir désérialiser ce modèle (sinon Keras 3 échoue sur la
# classe 'Functional' du format legacy).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler

from models.ML_classique.train_lgbm import _extract_features as _lgbm_extract_features
from models.Reservoir_Computing.MESN import (
    SENSOR_SPECS_CSV,
    _has_min_coverage,
    _reservoir_states,
    _sensor_inputs,
    _sensor_window,
    _summarize_states,
    _window_ms,
    _ReadoutProbaWrapper,
    load_sessions_from_csv,
)

TARGET_NAMES = ["baseline", "activity", "pre_fatigue", "fatigue"]
NUM_CLASSES = 4
OUTPUT_PATH = PROCESSED_DIR / "teacher_soft_labels.npz"


# ══════════════════════════════════════════════════════════════════════
# 1. FENÊTRES STUDENT DE RÉFÉRENCE
# ══════════════════════════════════════════════════════════════════════
def _extract_student_windows(df_labeled):
    """
    Reproduit exactement l'indexation par glissement de
    semi_supervised.extract_windows (même range(0, len-W+1, S) par session
    triée) pour que les fenêtres produites ici soient identiques à celles que
    train_cnn1.py utiliserait — mais conserve en plus le timestamp de départ
    de chaque fenêtre, nécessaire pour interroger LGBM/MESN sur le MÊME
    intervalle temporel (extract_windows ne renvoie que (X, y), pas les
    métadonnées de fenêtre).
    """
    config = WINDOW_CONFIGS["default"]
    w_size, s_size = config["window_size"], config["step_size"]
    window_ms, _ = _window_ms(None)

    df_sorted = df_labeled.sort_values([COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP])
    entries = []
    for (participant, session), group in df_sorted.groupby([COL_PARTICIPANT, COL_SESSION]):
        X_raw = group[SIGNAL_COLS].values.astype(np.float32)
        y_raw = group[COL_LABEL].values
        ts = group[COL_TIMESTAMP].values.astype(np.float64)
        for start in range(0, len(X_raw) - w_size + 1, s_size):
            end = start + w_size
            vals, counts = np.unique(y_raw[start:end], return_counts=True)
            start_ms = float(ts[start])
            entries.append({
                "participant": int(participant),
                "session": int(session),
                "start_ms": start_ms,
                "end_ms": start_ms + window_ms,
                "y_true": int(vals[np.argmax(counts)]),
                "X_window": X_raw[start:end],
            })
    return entries, w_size, s_size


# ══════════════════════════════════════════════════════════════════════
# 2. ADAPTATION MESN — PRÉDICTION SUR UN INTERVALLE ARBITRAIRE
# ══════════════════════════════════════════════════════════════════════
def _mesn_predict_interval(session, start_ms, end_ms, reservoirs, scalers, mesn_params, sensor_specs):
    """
    Adaptation minimale de MESN.transform_session_windows pour évaluer UN
    intervalle (start_ms, end_ms) donné par la grille CNN_1D, au lieu du
    découpage natif MESN (propre fenêtre/pas dérivés de _window_ms). Réutilise
    telles quelles les briques de MESN.py (_sensor_window, _has_min_coverage,
    _sensor_inputs qui gère déjà le padding/troncature à la taille nominale,
    _reservoir_states, _summarize_states) sans les dupliquer ni les modifier.

    Retourne None si un capteur est manquant ou sous la couverture minimale
    (cf _has_min_coverage) : la fenêtre est alors écartée par l'appelant,
    jamais imputée.
    """
    window_ms = end_ms - start_ms
    washout = int(mesn_params.get("washout", 0))
    summary = mesn_params.get("state_summary", "last_mean_std")

    features = []
    for sensor_name in reservoirs.keys():
        df_sensor = session.sensors.get(sensor_name)
        value_cols = list(sensor_specs[sensor_name]["cols"].values())
        missing_cols = [f"{c}_missing" for c in value_cols]
        required_cols = value_cols + missing_cols
        if df_sensor is None or not all(c in df_sensor.columns for c in required_cols):
            return None

        x_sensor = _sensor_window(df_sensor, start_ms, end_ms, value_cols)
        if not _has_min_coverage(sensor_name, x_sensor, window_ms, mesn_params, sensor_specs=sensor_specs):
            return None

        x_input = _sensor_inputs(df_sensor, sensor_name, start_ms, end_ms, scalers, sensor_specs=sensor_specs)
        states = _reservoir_states(x_input, reservoirs[sensor_name], washout)
        features.append(_summarize_states(states, summary))

    return np.concatenate(features).astype(np.float32)


def _load_mesn_params():
    """
    Charge les hyperparamètres (washout, state_summary, min_coverage) réellement
    utilisés pour entraîner MESN_global.pkl. reservoirs/readout/scalers sont
    déjà "figés" dans le pickle (les objets Reservoir/Ridge sont chargés tels
    quels) ; seuls ces 3 scalaires sont nécessaires pour rejouer l'inférence
    sur un intervalle (cf _mesn_predict_interval). Sauvegardés dans
    metrics.json["MESN"]["params"] par MESN.py::main() lors du dernier
    entraînement global.
    """
    defaults = {"washout": 0, "state_summary": "last_mean_std", "min_coverage": 0.35}
    if not METRICS_PATH.exists():
        print(f"  [WARN] {METRICS_PATH} introuvable — repli sur params MESN par défaut : {defaults}")
        return defaults
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    params = metrics.get("MESN", {}).get("params")
    if not params:
        print(f"  [WARN] metrics.json ne contient pas MESN.params — repli par défaut : {defaults}")
        return defaults
    return params


# ══════════════════════════════════════════════════════════════════════
# 3. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("ÉTAPE 1 — SOFT LABELS TEACHER (LGBM + MESN + CNN_LSTM) POUR LE STUDENT CNN_1D")
    print("=" * 70)

    if not DATA_MODEL_READY.exists():
        return print(f"Dataset non trouvé : {DATA_MODEL_READY}")
    if not OUTPUT_PATH_NO_SMOTE_FREQ_NO_SYNC.exists():
        return print(f"Dataset MESN (raw, non sync) non trouvé : {OUTPUT_PATH_NO_SMOTE_FREQ_NO_SYNC}")

    # ---- Dataset complet (toutes sessions), pas de split LOSO ici ----
    df = pd.read_csv(DATA_MODEL_READY)
    df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors="coerce").fillna(-1).astype(int)
    df_labeled = df[df[COL_LABEL] >= 0].copy()
    n_sessions = df_labeled[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().shape[0]
    print(f"Dataset labellisé : {len(df_labeled)} lignes, {n_sessions} sessions")

    # ---- Fenêtres student de référence ----
    entries, w_size, s_size = _extract_student_windows(df_labeled)
    n_candidates = len(entries)
    print(f"Fenêtres candidates (grille CNN_1D, window={w_size} step={s_size}) : {n_candidates}")
    if n_candidates == 0:
        return print("Aucune fenêtre candidate — abandon.")

    # ---- Chargement des 3 teachers, tels quels (aucun réentraînement) ----
    print("\nChargement des teachers (artefacts déjà entraînés)...")
    lgbm_model = joblib.load(MODELS_DIR / "LGBM" / "LGBM_global.pkl")
    print(f"  LGBM  : {MODELS_DIR / 'LGBM' / 'LGBM_global.pkl'}")

    mesn_bundle = joblib.load(MODELS_DIR / "MESN" / "MESN_global.pkl")
    mesn_reservoirs = mesn_bundle["reservoirs"]
    mesn_scalers = mesn_bundle["scalers"]
    mesn_proba_wrapper = _ReadoutProbaWrapper(mesn_bundle["readout"])
    mesn_params = _load_mesn_params()
    print(f"  MESN  : {MODELS_DIR / 'MESN' / 'MESN_global.pkl'}  "
          f"(washout={mesn_params.get('washout')}, state_summary={mesn_params.get('state_summary')}, "
          f"min_coverage={mesn_params.get('min_coverage'):.3f})")

    cnn_lstm_model = tf.keras.models.load_model(MODELS_DIR / "CNN-LSTM" / "CNN_LSTM_global_float32.keras")
    print(f"  CNN_LSTM : {MODELS_DIR / 'CNN-LSTM' / 'CNN_LSTM_global_float32.keras'}")

    # NOTE IMPORTANTE : le RobustScaler exact utilisé à l'entraînement du modèle
    # global CNN_LSTM (train_cnn1_lstm.py::train_global_model) n'a jamais été
    # persisté sur disque — seul le .keras est sauvegardé. On refit donc ici un
    # RobustScaler sur l'intégralité du dataset labellisé (mêmes colonnes,
    # même distribution globale) comme proxy le plus proche disponible sans
    # réentraîner ce teacher. Approximation documentée : si l'accuracy CNN_LSTM
    # ci-dessous semble anormalement basse par rapport à edge_comparison dans
    # metrics.json, ce refit de scaler est le premier suspect à vérifier.
    cnn_lstm_scaler = RobustScaler().fit(df_labeled[SIGNAL_COLS].values)

    # ---- Sessions MESN : signal brut par capteur, non synchronisé ----
    print("\nChargement des sessions MESN (signal brut par capteur, non sync)...")
    mesn_sessions = load_sessions_from_csv(OUTPUT_PATH_NO_SMOTE_FREQ_NO_SYNC, sensor_specs=SENSOR_SPECS_CSV)
    mesn_sessions_by_key = {(s.participant, s.session): s for s in mesn_sessions}
    print(f"  Sessions MESN disponibles : {len(mesn_sessions_by_key)}")

    # ---- Boucle d'alignement : LGBM + MESN fenêtre par fenêtre ----
    print(f"\nCalcul des prédictions teacher sur {n_candidates} fenêtres candidates...")
    t0 = time.time()

    valid_entries = []
    proba_lgbm_list, proba_mesn_list = [], []
    n_lost_missing_session = 0
    n_lost_mesn_coverage = 0

    for i, entry in enumerate(entries):
        key = (entry["participant"], entry["session"])
        session = mesn_sessions_by_key.get(key)
        if session is None:
            n_lost_missing_session += 1
            continue

        feat_mesn = _mesn_predict_interval(
            session, entry["start_ms"], entry["end_ms"],
            mesn_reservoirs, mesn_scalers, mesn_params, SENSOR_SPECS_CSV,
        )
        if feat_mesn is None:
            n_lost_mesn_coverage += 1
            continue

        feat_lgbm = _lgbm_extract_features(entry["X_window"])
        proba_l = lgbm_model.predict_proba(feat_lgbm.reshape(1, -1))[0]
        proba_m = mesn_proba_wrapper.predict_proba(feat_mesn.reshape(1, -1))[0]

        valid_entries.append(entry)
        proba_lgbm_list.append(proba_l)
        proba_mesn_list.append(proba_m)

        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{n_candidates} fenêtres examinées ({time.time() - t0:.0f}s)")

    print(f"  Alignement LGBM/MESN terminé en {time.time() - t0:.0f}s")

    n_valid = len(valid_entries)
    n_lost = n_candidates - n_valid
    pct_lost = 100.0 * n_lost / n_candidates

    # ---- CNN_LSTM en batch (uniquement sur les fenêtres déjà valides) ----
    if n_valid > 0:
        X_cnn_lstm = np.stack([e["X_window"] for e in valid_entries]).astype(np.float32)
        n, w, f = X_cnn_lstm.shape
        X_cnn_lstm_scaled = cnn_lstm_scaler.transform(X_cnn_lstm.reshape(-1, f)).reshape(n, w, f)
        proba_cnn_lstm_arr = np.asarray(cnn_lstm_model.predict(X_cnn_lstm_scaled, verbose=0))
    else:
        proba_cnn_lstm_arr = np.zeros((0, NUM_CLASSES), dtype=np.float32)

    proba_lgbm_arr = np.stack(proba_lgbm_list) if proba_lgbm_list else np.zeros((0, NUM_CLASSES), dtype=np.float32)
    proba_mesn_arr = np.stack(proba_mesn_list) if proba_mesn_list else np.zeros((0, NUM_CLASSES), dtype=np.float32)
    soft_label = ((proba_lgbm_arr + proba_mesn_arr + proba_cnn_lstm_arr) / 3.0).astype(np.float32)

    y_true_arr = np.array([e["y_true"] for e in valid_entries], dtype=np.int32)
    participant_arr = np.array([e["participant"] for e in valid_entries], dtype=np.int32)
    session_arr = np.array([e["session"] for e in valid_entries], dtype=np.int32)
    start_ms_arr = np.array([e["start_ms"] for e in valid_entries], dtype=np.float64)
    end_ms_arr = np.array([e["end_ms"] for e in valid_entries], dtype=np.float64)
    X_cnn1d_window_arr = (
        np.stack([e["X_window"] for e in valid_entries]).astype(np.float32)
        if n_valid > 0 else np.zeros((0, w_size, len(SIGNAL_COLS)), dtype=np.float32)
    )

    # ---- Sauvegarde ----
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        participant=participant_arr,
        session=session_arr,
        start_ms=start_ms_arr,
        end_ms=end_ms_arr,
        y_true=y_true_arr,
        X_cnn1d_window=X_cnn1d_window_arr,
        soft_label=soft_label,
        proba_lgbm=proba_lgbm_arr.astype(np.float32),
        proba_mesn=proba_mesn_arr.astype(np.float32),
        proba_cnn_lstm=proba_cnn_lstm_arr.astype(np.float32),
        signal_cols=np.array(SIGNAL_COLS),
        window_size=np.array(w_size),
        step_size=np.array(s_size),
    )
    print(f"\nSoft labels sauvegardés : {OUTPUT_PATH}  "
          f"({OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    # ---- Résumé (étape 1g) ----
    print("\n" + "=" * 70)
    print("RÉSUMÉ — ALIGNEMENT & QUALITÉ DE L'ENSEMBLE TEACHER")
    print("=" * 70)
    print(f"Fenêtres candidates (grille CNN_1D)       : {n_candidates}")
    print(f"  perdues — session MESN absente          : {n_lost_missing_session}")
    print(f"  perdues — couverture MESN insuffisante   : {n_lost_mesn_coverage}")
    print(f"Fenêtres valides (3 teachers OK)          : {n_valid}")
    print(f"Taux de perte à l'alignement               : {pct_lost:.1f}%")

    if pct_lost > 20.0:
        print("\n" + "!" * 70)
        print(f"ALERTE : taux de perte {pct_lost:.1f}% > 20% — possible problème")
        print("d'alignement plus profond à corriger AVANT de lancer l'étape 2.")
        print("!" * 70)

    if n_valid == 0:
        print("\nAucune fenêtre valide — impossible d'évaluer l'ensemble teacher.")
        return

    y_pred_lgbm = np.argmax(proba_lgbm_arr, axis=1)
    y_pred_mesn = np.argmax(proba_mesn_arr, axis=1)
    y_pred_cnn_lstm = np.argmax(proba_cnn_lstm_arr, axis=1)
    y_pred_ensemble = np.argmax(soft_label, axis=1)

    def _report(name, y_pred):
        acc = accuracy_score(y_true_arr, y_pred)
        f1m = f1_score(y_true_arr, y_pred, average="macro", zero_division=0)
        bal = balanced_accuracy_score(y_true_arr, y_pred)
        print(f"  {name:<10} accuracy={acc:.4f}  f1_macro={f1m:.4f}  balanced_acc={bal:.4f}")
        return acc

    print("\nPerformance par teacher (sur les fenêtres valides, vs y_true) :")
    acc_lgbm = _report("LGBM", y_pred_lgbm)
    acc_mesn = _report("MESN", y_pred_mesn)
    acc_cnn_lstm = _report("CNN_LSTM", y_pred_cnn_lstm)
    acc_ensemble = _report("Ensemble", y_pred_ensemble)

    best_single = max(acc_lgbm, acc_mesn, acc_cnn_lstm)
    best_name = ["LGBM", "MESN", "CNN_LSTM"][int(np.argmax([acc_lgbm, acc_mesn, acc_cnn_lstm]))]
    print(f"\nMeilleur teacher individuel : {best_name} ({best_single:.4f})")
    print(f"Ensemble (soft_label → argmax) : {acc_ensemble:.4f}")

    if acc_ensemble <= best_single:
        print("\n" + "!" * 70)
        print("ARRÊT RECOMMANDÉ : l'ensemble teacher n'est PAS meilleur que le")
        print(f"meilleur teacher individuel ({best_name}). Un ensemble qui n'apporte")
        print("rien ne vaut pas la peine d'être distillé — ne pas lancer l'étape 2")
        print("tel quel sans revoir la composition de l'ensemble ou l'alignement.")
        print("!" * 70)
    else:
        print(f"\nGain ensemble vs meilleur teacher : +{(acc_ensemble - best_single) * 100:.2f} pts")
        print("L'ensemble teacher est meilleur que chaque teacher individuel — l'étape 2 peut être lancée.")


if __name__ == "__main__":
    main()
