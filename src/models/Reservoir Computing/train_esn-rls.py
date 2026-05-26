import gc
import importlib.util
import json
import random
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from reservoirpy.nodes import RLS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import *

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

_BASE_SPEC = importlib.util.spec_from_file_location(
    "train_esn_base",
    Path(__file__).resolve().with_name("train_esn.py"),
)
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


MODEL_NAME = "ESN_RLS"
LABEL_MAPPING = {-1: 0, 0: 1, 1: 2}
TARGET_NAMES = ["baseline (-1)", "activity (0)", "fatigue (1)"]
RLS_FEATURE_CLIP = 8.0


# ============================================================================
# 1. OUTILS GENERAUX
# ============================================================================
def _free_memory(*objects):
    """
    Libere explicitement les objets temporaires et force le garbage collector.
    Utile pendant les folds LOSO et les essais Optuna pour limiter l'usage RAM.
    """
    for obj in objects:
        del obj
    gc.collect()


# ============================================================================
# 2. READOUT RLS ET PRETRAITEMENT DES FEATURES
# ============================================================================
def build_model(params):
    """
    Cree le readout RLS qui apprend a classifier les features produites par l'ESN.
    Le Reservoir est construit dans train_esn.py et seul le readout change ici.
    """
    return RLS(
        alpha=float(params["rls_alpha"]),
        forgetting=float(params.get("forgetting", 1.0)),
        output_dim=len(LABEL_MAPPING),
        name="readout_rls",
    )


def _to_onehot(y, num_classes=None):
    """
    Encode les labels numeriques en vecteurs One-Hot pour l'apprentissage du RLS.
    """
    num_classes = num_classes or len(LABEL_MAPPING)
    y_onehot = np.zeros((len(y), num_classes), dtype=np.float32)
    y_onehot[np.arange(len(y)), y.astype(int)] = 1.0
    return y_onehot


def _clean_features(x):
    """
    Nettoie les features ESN avant le RLS : remplace NaN/inf et limite les valeurs extremes.
    Cette protection evite les instabilites numeriques du readout recursif.
    """
    x = np.nan_to_num(x, nan=0.0, posinf=RLS_FEATURE_CLIP, neginf=-RLS_FEATURE_CLIP)
    return np.clip(x, -RLS_FEATURE_CLIP, RLS_FEATURE_CLIP).astype(np.float32)


def fit_feature_scaler(x_train):
    """
    Ajuste un StandardScaler sur les features d'entrainement et renvoie les features nettoyees.
    Le scaling se fait apres l'extraction ESN, donc sur les etats resumes du reservoir.
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    return scaler, _clean_features(x_scaled)


def transform_features(feature_scaler, x):
    """
    Applique le scaler deja appris aux features de validation/test puis les nettoie.
    """
    return _clean_features(feature_scaler.transform(x))


def fit_readout(model, x_train, y_train):
    """
    Entraine le readout RLS sur les features ESN.
    Les classes sont ponderees pour compenser le desequilibre du dataset.
    """
    # La racine carree des poids permet d'appliquer la ponderation sur X et y.
    weights = np.sqrt(base._sample_weights(y_train)).reshape(-1, 1)
    x_train = _clean_features(x_train)
    x_weighted = x_train * weights
    y_weighted = _to_onehot(y_train) * weights

    # Les RuntimeWarning sont transformes en erreurs pour detecter les divergences RLS.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        model.fit(x_weighted, y_weighted)

    # Verification finale : un poids non fini rendrait toutes les predictions invalides.
    if not np.all(np.isfinite(model.Wout)):
        raise FloatingPointError("RLS Wout contient des valeurs non finies")
    if not np.all(np.isfinite(model.bias)):
        raise FloatingPointError("RLS bias contient des valeurs non finies")
    return model


def predict_readout(model, x):
    """
    Calcule les scores du RLS puis retourne la classe avec le score maximal.
    """
    scores = np.asarray(model.run(_clean_features(x)), dtype=np.float32)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("RLS predictions contiennent des valeurs non finies")
    # Argmax sur les sorties One-Hot apprises : 0=baseline, 1=activity, 2=fatigue.
    return np.argmax(scores, axis=1).astype(np.int32)


# ============================================================================
# 3. OPTUNA
# ============================================================================
def _suggest_esn_rls_params(trial):
    """
    Definit l'espace de recherche Optuna pour le couple ESN + RLS.
    Les hyperparametres ESN controlent le reservoir, ceux du RLS controlent le readout.
    """
    space = ESN_RLS_OPTUNA_SPACE
    return {
        "n_reservoir": trial.suggest_categorical("n_reservoir", space["n_reservoir"]),
        "spectral_radius": trial.suggest_float("spectral_radius", **space["spectral_radius"]),
        "sparsity": trial.suggest_float("sparsity", **space["sparsity"]),
        "leak_rate": trial.suggest_float("leak_rate", **space["leak_rate"]),
        "input_scaling": trial.suggest_float("input_scaling", **space["input_scaling"]),
        "rls_alpha": trial.suggest_float("rls_alpha", **space["rls_alpha"]),
        "forgetting": trial.suggest_float("forgetting", **space["forgetting"]),
        "washout": trial.suggest_categorical("washout", space["washout"]),
        "state_summary": trial.suggest_categorical("state_summary", space["state_summary"]),
        "random_state": 42,
    }


def optuna_objective(trial, df, sessions, window_size, step_size):
    """
    Evalue un jeu d'hyperparametres Optuna sur quelques sessions de validation.
    Chaque session est retiree de l'entrainement puis utilisee comme validation temporaire.
    """
    params = _suggest_esn_rls_params(trial)
    scores = []

    for val_part, val_sess in sessions:
        model = None
        try:
            # Split session-wise : la session de validation ne fuit pas dans l'entrainement.
            df_train = df[~((df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess))].copy()
            df_val = df[(df[COL_PARTICIPANT] == val_part) & (df[COL_SESSION] == val_sess)].copy()

            # Normalisation des signaux bruts avant passage dans le reservoir ESN.
            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_val[SIGNAL_COLS] = scaler.transform(df_val[SIGNAL_COLS])

            # Extraction des features ESN avec la logique commune du script train_esn.py.
            reservoir = base._build_reservoir(params, len(SIGNAL_COLS))
            x_train, y_train = base.extract_esn_windows(df_train, reservoir, window_size, step_size)
            x_val, y_val = base.extract_esn_windows(df_val, reservoir, window_size, step_size)
            if len(y_train) == 0 or len(y_val) == 0:
                scores.append(0.0)
                continue
            # Deuxieme scaling : il stabilise les etats du reservoir avant le RLS.
            feature_scaler, x_train = fit_feature_scaler(x_train)
            x_val = transform_features(feature_scaler, x_val)

            model = build_model(params)
            fit_readout(model, x_train, y_train)
            y_pred = predict_readout(model, x_val)
            scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        except Exception as exc:
            print(f"  [WARN] trial split echoue ({type(exc).__name__}: {exc})")
            scores.append(0.0)
        finally:
            _free_memory(model)

    return float(np.mean(scores)) if scores else 0.0


def optimize_hyperparams(df, n_trials=ESN_RLS_OPTUNA_TRIALS):
    """
    Lance l'etude Optuna et sauvegarde les meilleurs hyperparametres ESN_RLS.
    """
    print(f"\nOPTUNA ESN_RLS - {n_trials} trials | {ESN_RLS_OPTUNA_SESSIONS} sessions\n" + "=" * 60)
    random.seed(42)
    window_size = WINDOW_CONFIGS["default"]["window_size"]
    step_size = WINDOW_CONFIGS["default"]["step_size"]
    # Liste des sessions disponibles pour construire un petit set de validation Optuna.
    all_sessions = [
        tuple(map(int, x))
        for x in df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values
    ]
    val_sessions = random.sample(all_sessions, min(ESN_RLS_OPTUNA_SESSIONS, len(all_sessions)))
    print(f"Sessions Optuna: {val_sessions}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(
        # Chaque trial mesure le F1 macro moyen sur les sessions de validation choisies.
        lambda trial: optuna_objective(trial, df, val_sessions, window_size, step_size),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
        gc_after_trial=True,
        catch=(Exception,),
    )

    print(f"\nBest F1-Macro: {study.best_value:.4f}")
    print(f"Best params  : {study.best_params}")
    # Persistance des resultats pour reutiliser les meilleurs parametres sans relancer Optuna.
    OPTUNA_PATH_ESN_RLS.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTUNA_PATH_ESN_RLS, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=4)
    return {**MODEL_PARAMS["ESN_RLS"], **study.best_params}


# ============================================================================
# 4. VISUALISATION
# ============================================================================
def plot_fold(fold_idx, test_part, test_sess, save_dir, f1_mac, bal_acc, y_true, y_pred, states):
    """
    Genere les graphiques d'un fold : scores, matrice de confusion et energie du reservoir.
    """
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        f"ESN_RLS | Fold {fold_idx} - P{test_part} S{test_sess}"
        f" | F1-Macro: {f1_mac:.3f} | Bal.Acc: {bal_acc:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    # Resume visuel des deux metriques principales du fold.
    axes[0].bar(["F1-Macro", "Balanced Acc"], [f1_mac, bal_acc], color=["#2563EB", "#059669"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Scores du fold")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Matrice de confusion limitee aux classes presentes dans y_true/y_pred.
    cm = confusion_matrix(y_true, y_pred)
    labels = [TARGET_NAMES[i] for i in np.unique(np.concatenate([y_true, y_pred]))]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        linewidths=0.5,
        linecolor="gray",
    )
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predit")
    axes[1].set_ylabel("Reel")
    axes[1].tick_params(axis="x", rotation=20)

    if states is not None and len(states) > 0:
        # Energie moyenne des etats internes : utile pour voir si le reservoir reste stable.
        energy = np.mean(np.abs(states), axis=1)
        axes[2].plot(np.arange(len(energy)), energy, color="#EA580C", linewidth=2)
        axes[2].set_title("Energie moyenne du reservoir")
        axes[2].set_xlabel("Temps apres washout")
        axes[2].set_ylabel("mean(|state|)")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    plot_path = save_dir / f"fold{fold_idx:02d}_P{test_part}_S{test_sess}_curves.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Courbes sauvegardees: {plot_path}")


# ============================================================================
# 5. MODELE GLOBAL
# ============================================================================
def train_global_model(df, best_params):
    """
    Entraine un modele ESN_RLS final sur tout le dataset et le sauvegarde avec ses scalers.
    Ce modele global sert ensuite a l'inference ou a l'embarquement edge.
    """
    print("\n" + "=" * 60 + f"\nMODELE GLOBAL - {MODEL_NAME}\n" + "=" * 60)
    config = WINDOW_CONFIGS["default"]
    window_size = config["window_size"]
    step_size = config["step_size"]

    # Normalisation des signaux bruts sur toutes les donnees disponibles.
    df_all = df.copy()
    scaler = RobustScaler()
    df_all[SIGNAL_COLS] = scaler.fit_transform(df_all[SIGNAL_COLS])

    # Extraction ESN sur tout le dataset avant l'entrainement du readout RLS final.
    reservoir = base._build_reservoir(best_params, len(SIGNAL_COLS))
    x_all, y_all = base.extract_esn_windows(df_all, reservoir, window_size, step_size)
    if len(y_all) == 0:
        print("  Aucun exemple global disponible.")
        return
    feature_scaler, x_all = fit_feature_scaler(x_all)

    model = build_model(best_params)
    fit_readout(model, x_all, y_all)

    models_dir = MODELS_DIR / MODEL_NAME
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{MODEL_NAME}_global.pkl"
    # Le pickle contient le readout, le reservoir et les scalers necessaires a l'inference.
    joblib.dump(
        {
            "model_name": MODEL_NAME,
            "readout": model,
            "reservoir": reservoir,
            "scaler": scaler,
            "feature_scaler": feature_scaler,
            "signal_cols": SIGNAL_COLS,
            "label_mapping": LABEL_MAPPING,
            "target_names": TARGET_NAMES,
            "window_size": window_size,
            "step_size": step_size,
            "params": best_params,
        },
        model_path,
    )
    print(f"  Modele global sauvegarde: {model_path}")


# ============================================================================
# 6. ENTRAINEMENT LOSO
# ============================================================================
def main():
    """
    Point d'entree du script : charge les donnees, entraine en LOSO, sauvegarde les metriques.
    LOSO signifie que chaque fold garde une session participant comme test.
    """
    print("\n" + "=" * 60 + f"\nTRAIN LOSO - {MODEL_NAME}\n" + "=" * 60)
    if not DATA_MODEL_READY.exists():
        return print(f"Dataset non trouve: {DATA_MODEL_READY}")

    df = pd.read_csv(DATA_MODEL_READY)
    # Mapping des labels originaux (-1, 0, 1) vers des indices compatibles avec One-Hot.
    df[COL_LABEL] = df[COL_LABEL].map(LABEL_MAPPING)

    # Chaque couple participant/session devient un fold de test.
    unique_sessions = [
        tuple(map(int, x))
        for x in df[df[COL_PARTICIPANT] >= 1][[COL_PARTICIPANT, COL_SESSION]]
        .drop_duplicates()
        .values
    ]

    best_params = optimize_hyperparams(df) if USE_OPTUNA_ESN_RLS else MODEL_PARAMS["ESN_RLS"]
    best_params = {**MODEL_PARAMS["ESN_RLS"], **best_params}

    all_metrics = []
    plots_dir = BASE_DIR / "training_curves" / MODEL_NAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (test_part, test_sess) in enumerate(unique_sessions, start=1):
        print(f"\n--- FOLD {fold_idx}/{len(unique_sessions)} : Test P{test_part} S{test_sess} ---")
        config = WINDOW_CONFIGS.get(test_part, WINDOW_CONFIGS["default"])
        window_size = config["window_size"]
        step_size = config["step_size"]

        # Split LOSO : la session test est completement exclue de l'entrainement.
        df_train = df[~((df[COL_PARTICIPANT] == test_part) & (df[COL_SESSION] == test_sess))].copy()
        df_test = df[(df[COL_PARTICIPANT] == test_part) & (df[COL_SESSION] == test_sess)].copy()

        model = None
        try:
            # RobustScaler limite l'effet des valeurs extremes dans les signaux capteurs.
            scaler = RobustScaler()
            df_train[SIGNAL_COLS] = scaler.fit_transform(df_train[SIGNAL_COLS])
            df_test[SIGNAL_COLS] = scaler.transform(df_test[SIGNAL_COLS])

            # Le reservoir extrait une representation temporelle fixe pour chaque fenetre.
            reservoir = base._build_reservoir(best_params, len(SIGNAL_COLS))
            x_train, y_train = base.extract_esn_windows(df_train, reservoir, window_size, step_size)
            x_test, y_test = base.extract_esn_windows(df_test, reservoir, window_size, step_size)

            if len(y_train) == 0 or len(y_test) == 0:
                print("  WARNING: train/test vide. Fold ignore.")
                continue
            # Normalisation des features ESN avant apprentissage/prediction RLS.
            feature_scaler, x_train = fit_feature_scaler(x_train)
            x_test = transform_features(feature_scaler, x_test)

            print(f"  Train: {len(x_train)} fenetres | Test: {len(x_test)} fenetres")

            model = build_model(best_params)
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

            # Etats de la premiere fenetre test pour visualiser la dynamique du reservoir.
            states = base.first_window_states(df_test, reservoir, window_size)
            plot_fold(
                fold_idx,
                test_part,
                test_sess,
                plots_dir,
                f1_mac,
                bal_acc,
                y_test,
                y_pred,
                states,
            )

            all_metrics.append(
                {
                    "Fold": fold_idx,
                    "Participant": int(test_part),
                    "Session": int(test_sess),
                    "F1_Macro": float(f1_mac),
                    "Balanced_Accuracy": float(bal_acc),
                }
            )

        except Exception as exc:
            print(f"  Fold {fold_idx} echoue ({type(exc).__name__}: {exc})")
        finally:
            _free_memory(model)

    if not all_metrics:
        return print("Aucun fold complete.")

    # Synthese des metriques sur tous les folds LOSO.
    df_res = pd.DataFrame(all_metrics)
    print("\n" + "=" * 60 + f"\nRESULTATS FINAUX - {MODEL_NAME}\n" + "=" * 60)
    print(df_res.describe().loc[["mean", "std"]])

    # Lecture defensive du metrics.json existant : s'il est vide/invalide, on repart proprement.
    curr_metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                content = f.read().strip()
            curr_metrics = json.loads(content) if content else {}
        except (json.JSONDecodeError, ValueError):
            print(f"metrics.json invalide ou vide, reinitialisation: {METRICS_PATH}")
            curr_metrics = {}

    # Ajout/remplacement des resultats ESN_RLS dans le fichier de metriques global.
    curr_metrics[MODEL_NAME] = {
        "mean_f1": float(df_res["F1_Macro"].mean()),
        "std_f1": float(df_res["F1_Macro"].std()),
        "mean_bal_acc": float(df_res["Balanced_Accuracy"].mean()),
        "std_bal_acc": float(df_res["Balanced_Accuracy"].std()),
        "params": best_params,
        "folds": all_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(curr_metrics, f, indent=4)
    print(f"\nMetriques -> {METRICS_PATH}")

    # Entrainement final sur toutes les donnees apres l'evaluation LOSO.
    train_global_model(df, best_params)


if __name__ == "__main__":
    main()
"""Train ESN-RLS on the dataset and evaluate it with LOSO cross-validation."""