"""
Étape 2 de la distillation de connaissance : entraîne le student CNN_1D
(poids aléatoires, ré-entraîné depuis zéro — build_model() réutilisée telle
quelle depuis train_cnn1.py) sur une loss combinée hard-label + soft-label
teacher (Hinton et al.), avec la même boucle LOSO (36 folds) que
train_cnn1.py, pour rester comparable au résultat de référence
(CNN_1D LOSO mean_f1 = 0.789 dans metrics.json).

DÉVIATION DOCUMENTÉE vs train_cnn1.py — PAS DE SMOTE ICI :
train_cnn1.py applique SMOTE (resample_dataframe) sur les LIGNES de df_fit
avant fenêtrage. Les fenêtres résultantes mélangent alors lignes réelles et
lignes synthétiques interpolées, et ne correspondent plus aux fenêtres
(participant, session, start_ms, end_ms) pour lesquelles l'étape 1 a calculé
un soft label teacher (les 3 teachers n'ont jamais vu ces fenêtres
synthétiques, et les ré-évaluer dessus n'aurait pas de sens : les lignes
SMOTE sont des interpolations du signal synchronisé CNN, pas des mesures
capteur réelles que MESN/LGBM pourraient évaluer sur leur propre grille).
Plutôt que d'inventer un soft label factice (= one-hot, ce qui diluerait
silencieusement le signal de distillation sur une fraction significative des
fenêtres d'entraînement), ce script restreint l'ensemble train/val de chaque
fold aux fenêtres PRÉCALCULÉES par generate_teacher_labels.py (donc 100% de
signal de distillation réel), et remplace SMOTE par un sample_weight balancé
par classe (compute_class_weight) sur la loss combinée. Conséquence : le
volume d'entraînement par fold est plus petit que celui de train_cnn1.py
(728 fenêtres valides au total, réparties sur 36 sessions, vs un ensemble
gonflé par SMOTE) — à garder en tête en comparant les F1 par fold.

Architecture : hyperparamètres CNN_1D déjà optimisés par Optuna
(MODEL_PARAMS["CNN_1D"]), réutilisés tels quels (pas de nouvelle recherche
d'architecture ici — seule la loss change, pour isoler l'effet de la
distillation).
"""
import gc
import json
import os
import sys
import warnings
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow.lite')
tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import *

from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.utils import to_categorical

from models.DL.train_cnn1 import (
    build_model,
    plot_fold,
    _compute_classification_metrics,
    _free_memory,
    _predict_tflite,
    _save_edge_comparison_metrics,
    _save_tflite_int8,
)
from models.semi_supervised import extract_all_windows
from models.distillation.generate_teacher_labels import OUTPUT_PATH as SOFT_LABELS_PATH

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAME = "CNN_1D_DISTILLED"
BASELINE_MODEL_NAME = "CNN_1D"
NUM_CLASSES = 4

# Hyperparamètres de distillation — POINT DE DÉPART, PAS DES VALEURS FIGÉES.
# alpha : poids donné au hard label (categorical_crossentropy) vs au soft
# label teacher (KL divergence). alpha=0.3 -> 70% du signal vient du teacher,
# comme demandé. À ajuster empiriquement (0.2-0.5 est la plage usuelle en KD).
ALPHA = 0.3
# Température de lissage softmax (Hinton et al.), appliquée à l'entraînement
# de la distillation uniquement, jamais à l'inférence normale. T=3 -> point de
# départ usuel. À ajuster empiriquement.
TEMPERATURE = 3.0


# ══════════════════════════════════════════════════════════════════════
# 1. TEMPÉRATURE SUR DES PROBABILITÉS (PAS DE LOGITS DISPONIBLES)
# ══════════════════════════════════════════════════════════════════════
def _soften_np(p, temperature, eps=1e-8):
    """
    Les 3 teachers n'exposent que des probabilités post-softmax (predict_proba),
    pas les logits bruts (LGBM/MESN n'en ont pas d'équivalent direct exploité
    ici). Mathématiquement, softmax(z/T) = p^(1/T) / sum(p^(1/T)) quand
    p = softmax(z) : appliquer cette transformation directement sur les
    probabilités équivaut donc exactement à diviser des logits z=log(p) par T
    avant softmax — pas une approximation, mais l'opération correcte étant
    donné ce qui est disponible.
    """
    p = np.clip(p, eps, 1.0)
    pt = p ** (1.0 / temperature)
    return pt / pt.sum(axis=-1, keepdims=True)


def _soften_tf(p, temperature, eps=1e-7):
    p = tf.clip_by_value(p, eps, 1.0)
    pt = tf.pow(p, 1.0 / temperature)
    return pt / tf.reduce_sum(pt, axis=-1, keepdims=True)


# ══════════════════════════════════════════════════════════════════════
# 2. LOSS COMBINÉE hard-label + distillation (Hinton et al.)
# ══════════════════════════════════════════════════════════════════════
def make_distill_loss(alpha, temperature, num_classes):
    """
    y_combined (target passé à model.fit) concatène [one_hot(y_true) | soft_label
    déjà tempéré par _soften_np] sur l'axe des classes (shape (batch, 2*num_classes))
    -- Keras ne permet pas nativement de passer deux targets différents à une
    seule loss, ce doublement de colonnes est le contournement standard.
    """
    def distill_loss(y_combined, y_pred):
        y_true_onehot = y_combined[:, :num_classes]
        soft_label_t = y_combined[:, num_classes:]
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        cce = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred)

        y_pred_t = _soften_tf(y_pred, temperature)
        # Facteur T^2 (Hinton et al.) : compense la réduction de magnitude du
        # gradient induite par le lissage en température côté KL.
        kl = tf.keras.losses.kullback_leibler_divergence(soft_label_t, y_pred_t) * (temperature ** 2)

        return alpha * cce + (1.0 - alpha) * kl

    return distill_loss


def hard_label_accuracy(y_combined, y_pred):
    y_true_onehot = y_combined[:, :NUM_CLASSES]
    return tf.keras.metrics.categorical_accuracy(y_true_onehot, y_pred)


# plot_fold() (importée de train_cnn1.py, non modifiée) lit en dur
# history.history["accuracy"]/["val_accuracy"]. Keras nomme la clé de
# l'historique d'après le nom de la fonction métrique passée à compile() —
# ici renommée "accuracy" pour que l'historique produise ces clés, sans
# toucher à train_cnn1.py ni changer ce que la métrique calcule (toujours
# l'accuracy sur le hard label, cf ci-dessus).
hard_label_accuracy.__name__ = "accuracy"


# ══════════════════════════════════════════════════════════════════════
# 3. CHARGEMENT DES SOFT LABELS (ÉTAPE 1)
# ══════════════════════════════════════════════════════════════════════
def _load_soft_label_windows():
    if not SOFT_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"{SOFT_LABELS_PATH} introuvable — lancez d'abord "
            f"src/models/distillation/generate_teacher_labels.py (étape 1)."
        )
    data = np.load(SOFT_LABELS_PATH, allow_pickle=False)
    saved_cols = [str(c) for c in data["signal_cols"]]
    if saved_cols != SIGNAL_COLS:
        raise ValueError(
            "SIGNAL_COLS a changé depuis la génération des soft labels — "
            "relancer generate_teacher_labels.py."
        )

    windows_by_session = {}
    n = len(data["participant"])
    for i in range(n):
        key = (int(data["participant"][i]), int(data["session"][i]))
        windows_by_session.setdefault(key, []).append({
            "y_true": int(data["y_true"][i]),
            "X": data["X_cnn1d_window"][i],
            "soft_label": data["soft_label"][i],
        })
    return windows_by_session, int(data["window_size"]), int(data["step_size"])


def _stack_entries(entries):
    X = np.stack([e["X"] for e in entries]).astype(np.float32)
    y = np.array([e["y_true"] for e in entries], dtype=np.int32)
    soft = np.stack([e["soft_label"] for e in entries]).astype(np.float32)
    return X, y, soft


def _scale_windows(X_raw, scaler):
    n, w, f = X_raw.shape
    return scaler.transform(X_raw.reshape(-1, f)).reshape(n, w, f).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 4. BOUCLE LOSO — COMPARAISON DISTILLÉ vs BASELINE
# ══════════════════════════════════════════════════════════════════════
def run_loso_comparison(df_labeled, windows_by_session, best_params):
    unique_sessions = [
        tuple(x) for x in
        df_labeled[df_labeled[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
    ]

    baseline_folds = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            baseline_metrics = json.load(f)
        for fold in baseline_metrics.get(BASELINE_MODEL_NAME, {}).get("folds", []):
            baseline_folds[(fold["Participant"], fold["Session"])] = fold["F1_Macro"]

    all_results = []
    import random
    random.seed(42)

    for test_idx, (test_part, test_sess) in enumerate(unique_sessions):
        print(f"\n--- FOLD {test_idx + 1}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")
        config = WINDOW_CONFIGS.get(test_part, WINDOW_CONFIGS["default"])
        W_SIZE, S_SIZE, EPOCHS = config["window_size"], config["step_size"], config["epochs"]

        df_pool = df_labeled[~((df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess))].copy()
        df_test = df_labeled[(df_labeled[COL_PARTICIPANT] == test_part) & (df_labeled[COL_SESSION] == test_sess)].copy()

        pool_sessions = [tuple(x) for x in df_pool[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values]
        val_part, val_sess = random.choice(pool_sessions)
        print(f"  Val : P{val_part} S{val_sess}")
        fit_sessions = [s for s in pool_sessions if s != (val_part, val_sess)]

        fit_entries = [w for s in fit_sessions for w in windows_by_session.get(s, [])]
        val_entries = windows_by_session.get((val_part, val_sess), [])

        if not fit_entries or not val_entries:
            print("  WARNING: pas assez de fenêtres teacher-validées (fit ou val vide). Fold ignoré.")
            continue

        # Scaler ajusté sur les lignes brutes du pool d'entraînement (fit,
        # hors val/test) — pas de SMOTE (cf docstring du module).
        df_fit_rows = df_pool[
            ~((df_pool[COL_PARTICIPANT] == val_part) & (df_pool[COL_SESSION] == val_sess))
        ]
        scaler = RobustScaler().fit(df_fit_rows[SIGNAL_COLS].values)

        X_fit_raw, y_fit, soft_fit = _stack_entries(fit_entries)
        X_val_raw, y_val, soft_val = _stack_entries(val_entries)
        X_fit = _scale_windows(X_fit_raw, scaler)
        X_val = _scale_windows(X_val_raw, scaler)

        df_test_scaled = df_test.copy()
        df_test_scaled[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])
        X_test, y_test = extract_all_windows(df_test_scaled, W_SIZE, S_SIZE)
        if len(X_test) == 0:
            print("  WARNING: test vide. Fold ignoré.")
            continue

        print(f"  Fit: {len(X_fit)} | Val: {len(X_val)} | Test: {len(X_test)} fenêtres "
              f"(teacher-validées, sans SMOTE)")

        soft_fit_t = _soften_np(soft_fit, TEMPERATURE)
        soft_val_t = _soften_np(soft_val, TEMPERATURE)
        y_fit_combined = np.concatenate([to_categorical(y_fit, NUM_CLASSES), soft_fit_t], axis=1)
        y_val_combined = np.concatenate([to_categorical(y_val, NUM_CLASSES), soft_val_t], axis=1)

        w = compute_class_weight("balanced", classes=np.unique(y_fit), y=y_fit)
        weight_dict = dict(zip(np.unique(y_fit), w))
        sample_weight_fit = np.array([weight_dict[yi] for yi in y_fit], dtype=np.float32)
        w_val = compute_class_weight("balanced", classes=np.unique(y_val), y=y_val)
        weight_dict_val = dict(zip(np.unique(y_val), w_val))
        sample_weight_val = np.array([weight_dict_val[yi] for yi in y_val], dtype=np.float32)

        model = None
        try:
            model = build_model(
                optuna.trial.FixedTrial(best_params),
                (W_SIZE, len(SIGNAL_COLS)), NUM_CLASSES
            )
            # Réutilise l'optimiseur déjà configuré par build_model (mêmes
            # hyperparamètres Optuna) — seule la loss change pour la distillation.
            model.compile(
                optimizer=model.optimizer,
                loss=make_distill_loss(ALPHA, TEMPERATURE, NUM_CLASSES),
                metrics=[hard_label_accuracy],
                jit_compile=False,
            )
            history = model.fit(
                X_fit, y_fit_combined,
                sample_weight=sample_weight_fit,
                epochs=EPOCHS,
                validation_data=(X_val, y_val_combined, sample_weight_val),
                batch_size=best_params.get("batch_size", 32),
                callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                           TerminateOnNaN()],
                verbose=1,
            )

            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            baseline_f1 = baseline_folds.get((test_part, test_sess))
            delta = (f1_mac - baseline_f1) if baseline_f1 is not None else None
            print(f"  F1-Macro distillé: {f1_mac:.4f} | Bal.Acc: {bal_acc:.4f}"
                  + (f" | F1-Macro baseline: {baseline_f1:.4f} | delta: {delta:+.4f}" if baseline_f1 is not None else ""))

            plots_dir = BASE_DIR / "training_curves" / "CNN_1D_DISTILLED"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_fold(history, test_idx + 1, test_part, test_sess,
                      plots_dir, f1_mac, bal_acc, y_test, y_pred)

            all_results.append({
                "Fold": test_idx + 1, "Participant": int(test_part), "Session": int(test_sess),
                "F1_Macro_Distilled": float(f1_mac), "Balanced_Accuracy_Distilled": float(bal_acc),
                "F1_Macro_Baseline": float(baseline_f1) if baseline_f1 is not None else None,
                "Delta": float(delta) if delta is not None else None,
                "N_Fit_Windows": int(len(X_fit)), "N_Val_Windows": int(len(X_val)),
            })

        except Exception as exc:
            print(f"  Fold {test_idx + 1} échoué ({type(exc).__name__}: {exc})")
        finally:
            _free_memory(model)
            del X_fit, X_val, X_test, y_fit, y_val, y_test
            gc.collect()

    return all_results


def _summarize_comparison(all_results):
    df_res = pd.DataFrame(all_results)
    paired = df_res.dropna(subset=["F1_Macro_Baseline"])

    print("\n" + "=" * 70)
    print("RÉSUMÉ COMPARATIF — CNN_1D DISTILLÉ vs BASELINE (LOSO)")
    print("=" * 70)
    print(f"Folds complétés : {len(df_res)}/36  |  Folds comparables (baseline dispo) : {len(paired)}")

    mean_distilled = float(df_res["F1_Macro_Distilled"].mean())
    std_distilled = float(df_res["F1_Macro_Distilled"].std())
    print(f"\nmean_f1 distillé  : {mean_distilled:.4f} ± {std_distilled:.4f}")

    summary = {
        "mean_f1_distilled": mean_distilled,
        "std_f1_distilled": std_distilled,
        "alpha": ALPHA,
        "temperature": TEMPERATURE,
        "params": None,  # rempli par l'appelant
        "folds": all_results,
    }

    if len(paired) >= 2:
        mean_baseline = float(paired["F1_Macro_Baseline"].mean())
        mean_delta = float(paired["Delta"].mean())
        n_improved = int((paired["Delta"] > 0).sum())
        print(f"mean_f1 baseline (folds appariés) : {mean_baseline:.4f}")
        print(f"delta moyen (distillé - baseline)  : {mean_delta:+.4f}")
        print(f"folds améliorés : {n_improved}/{len(paired)}")

        stat, p_value = wilcoxon(paired["F1_Macro_Distilled"], paired["F1_Macro_Baseline"])
        print(f"\nTest de Wilcoxon signé (distillé vs baseline, {len(paired)} paires) :")
        print(f"  statistique = {stat:.4f}  |  p-value = {p_value:.4f}")
        if p_value < 0.05:
            direction = "meilleur" if mean_delta > 0 else "moins bon"
            print(f"  -> différence statistiquement significative (p<0.05) : le student distillé est {direction}.")
        else:
            print("  -> différence NON statistiquement significative (p>=0.05).")

        summary["mean_f1_baseline_paired"] = mean_baseline
        summary["mean_delta"] = mean_delta
        summary["n_folds_improved"] = n_improved
        summary["wilcoxon_statistic"] = float(stat)
        summary["wilcoxon_p_value"] = float(p_value)
    else:
        print("Pas assez de folds appariés avec la baseline pour un test de Wilcoxon.")

    return summary


# ══════════════════════════════════════════════════════════════════════
# 5. MODÈLE GLOBAL DISTILLÉ + EXPORT TFLITE INT8 (même pipeline que CNN_1D)
# ══════════════════════════════════════════════════════════════════════
def train_distilled_global_model(df_labeled, windows_by_session, best_params):
    print("\n" + "=" * 70 + f"\nMODÈLE GLOBAL — {MODEL_NAME}\n" + "=" * 70)
    config = WINDOW_CONFIGS["default"]
    W_SIZE = config["window_size"]

    all_entries = [w for entries in windows_by_session.values() for w in entries]
    all_keys = list(windows_by_session.keys())
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(all_keys))
    split = int(0.9 * len(all_keys))
    train_sessions = {all_keys[i] for i in shuffled[:split]}
    val_sessions = {all_keys[i] for i in shuffled[split:]}

    train_entries = [w for k in train_sessions for w in windows_by_session[k]]
    val_entries = [w for k in val_sessions for w in windows_by_session[k]]
    print(f"  Fenêtres teacher-validées : train={len(train_entries)} val={len(val_entries)}")

    df_train_rows = df_labeled[
        df_labeled.set_index([COL_PARTICIPANT, COL_SESSION]).index.isin(train_sessions)
    ]
    scaler = RobustScaler().fit(df_train_rows[SIGNAL_COLS].values)

    X_tr_raw, y_tr, soft_tr = _stack_entries(train_entries)
    X_vl_raw, y_vl, soft_vl = _stack_entries(val_entries)
    X_tr = _scale_windows(X_tr_raw, scaler)
    X_vl = _scale_windows(X_vl_raw, scaler)

    soft_tr_t = _soften_np(soft_tr, TEMPERATURE)
    soft_vl_t = _soften_np(soft_vl, TEMPERATURE)
    y_tr_combined = np.concatenate([to_categorical(y_tr, NUM_CLASSES), soft_tr_t], axis=1)
    y_vl_combined = np.concatenate([to_categorical(y_vl, NUM_CLASSES), soft_vl_t], axis=1)

    w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    weight_dict = dict(zip(np.unique(y_tr), w))
    sample_weight_tr = np.array([weight_dict[yi] for yi in y_tr], dtype=np.float32)

    model = None
    try:
        model = build_model(optuna.trial.FixedTrial(best_params), (W_SIZE, len(SIGNAL_COLS)), NUM_CLASSES)
        model.compile(
            optimizer=model.optimizer,
            loss=make_distill_loss(ALPHA, TEMPERATURE, NUM_CLASSES),
            metrics=[hard_label_accuracy],
            jit_compile=False,
        )
        model.fit(
            X_tr, y_tr_combined,
            sample_weight=sample_weight_tr,
            validation_data=(X_vl, y_vl_combined),
            epochs=config["epochs"],
            batch_size=best_params.get("batch_size", 32),
            callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                       TerminateOnNaN()],
            verbose=1,
        )

        # Dossier dédié (pas MODELS_DIR / "CNN") pour ne pas partager
        # models_saved/CNN/ avec les artefacts du CNN_1D classique.
        models_dir = MODELS_DIR / "CNN_1D_DISTILLED"
        models_dir.mkdir(parents=True, exist_ok=True)
        stem = "CNN_1D_distilled_global"
        float32_path = models_dir / f"{stem}_float32.keras"
        model.save(float32_path)

        y_pred_float32 = np.argmax(model.predict(X_vl, verbose=0), axis=1)
        float32_metrics = _compute_classification_metrics(y_vl, y_pred_float32)
        float32_metrics["model_size_kb"] = float(float32_path.stat().st_size / 1024)

        X_repr = np.concatenate([X_tr, X_vl], axis=0)
        _save_tflite_int8(model, X_repr, models_dir, stem)
        tflite_path = models_dir / f"{stem}_int8.tflite"
        y_pred_tflite = _predict_tflite(tflite_path, X_vl)
        tflite_metrics = _compute_classification_metrics(y_vl, y_pred_tflite)
        tflite_metrics["model_size_kb"] = float(tflite_path.stat().st_size / 1024)

        edge_metrics = {"float32": float32_metrics, "int8_tflite": tflite_metrics}
        _save_edge_comparison_metrics(MODEL_NAME, edge_metrics)
        print(f"  Modèle global distillé exporté (.keras + .tflite + .h) — INT8 : "
              f"{tflite_metrics['model_size_kb']:.1f} KB / {STM32_FLASH_KB} KB budget flash.")
    except Exception as exc:
        print(f"  Modèle global distillé échoué ({type(exc).__name__}: {exc})")
        import traceback; traceback.print_exc()
    finally:
        _free_memory(model)
        del X_tr, X_vl, y_tr, y_vl
        gc.collect()


# ══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print(f"ÉTAPE 2 — ENTRAÎNEMENT DISTILLÉ DU STUDENT ({MODEL_NAME})")
    print(f"alpha={ALPHA}  temperature={TEMPERATURE}")
    print("=" * 70)

    if not DATA_MODEL_READY.exists():
        return print(f"Dataset non trouvé : {DATA_MODEL_READY}")

    df = pd.read_csv(DATA_MODEL_READY)
    df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors="coerce").fillna(-1).astype(int)
    df_labeled = df[df[COL_LABEL] >= 0].copy()

    windows_by_session, w_size, s_size = _load_soft_label_windows()
    n_total_windows = sum(len(v) for v in windows_by_session.values())
    print(f"Soft labels chargés : {n_total_windows} fenêtres teacher-validées "
          f"sur {len(windows_by_session)} sessions (window={w_size}, step={s_size})")

    best_params = MODEL_PARAMS["CNN_1D"]  # architecture déjà optimisée par Optuna, réutilisée telle quelle
    print(f"Architecture CNN_1D (réutilisée, pas de nouvelle recherche Optuna) : {best_params}")

    all_results = run_loso_comparison(df_labeled, windows_by_session, best_params)
    if not all_results:
        return print("Aucun fold complété.")

    summary = _summarize_comparison(all_results)
    summary["params"] = best_params

    curr_metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            content = f.read().strip()
        curr_metrics = json.loads(content) if content else {}
    curr_metrics[MODEL_NAME] = summary
    with open(METRICS_PATH, "w") as f:
        json.dump(curr_metrics, f, indent=4)
    print(f"\nRésumé comparatif → {METRICS_PATH} (clé '{MODEL_NAME}')")

    train_distilled_global_model(df_labeled, windows_by_session, best_params)


if __name__ == "__main__":
    main()
