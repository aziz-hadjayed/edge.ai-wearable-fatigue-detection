# --------------------------------------------------------------------// Importations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import RobustScaler

from config import *

# --------------------------------------------------------------------// Fonctions de préparation des données

PATH_BREATHING = "chest_raw_breathing.csv"
PATH_MARKERS = "exp_markers.csv"

# ==============================================================================
# Paramètres respiration
# ==============================================================================
FS_BREATHING = 25.0
FMIN, FMAX = 0.05, 0.80
MIN_DIST_S = 1.5
PROM_MIN = 100
PROM_OK = 500

# ==============================================================================
# Fréquences capteurs
# ==============================================================================

TARGET_FREQ = 4
TARGET_PERIOD = 1000 / TARGET_FREQ  # ms



keys_list = list(LABEL_MAP.keys())  # ["baseline", "activity", "fatigue"]


def get_label_intervals(markers_path):
    """
    Extrait les intervalles de temps pour chaque label d'intérêt depuis le fichier des marqueurs.
    Convertit les timestamps absolus en temps relatif par rapport au début de la session.
    """
    """
    Extrait les intervalles de temps pour chaque label d'intérêt depuis le fichier des marqueurs.
    Convertit les timestamps absolus en temps relatif par rapport au début de la session.
    """
    df = pd.read_csv(markers_path)
    df.columns = df.columns.str.strip()

    intervals = []
    active = {}

    for _, row in df.iterrows():
        ts = float(row["utcTime"])
        event = str(row["eventMarker"]).lower()

        for lbl in keys_list:
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


# ==============================================================================
# 2. Resample → garder temps relatif
# ==============================================================================
def resample_to_target_freq(df, native_freq, t_ref):
    """
    Ré-échantillonne un DataFrame à la fréquence cible (TARGET_FREQ).
    Aligne les timestamps par rapport à un t_ref absolu.
    Applique la méthode appropriée (mean, ffill, interpolate) selon la fréquence d'origine.
    """
    df = df.copy()

    # 🔥 Synchronisation : rendre les timestamps relatifs à t_ref (absolu)
    # AVANT resampling pour que tous les capteurs soient alignés sur le même référentiel
    df["timestamp"] = df["timestamp"] - t_ref

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms").astype("datetime64[us]")
    df = df.set_index("datetime").drop(columns=["timestamp"])

    period = f"{int(TARGET_PERIOD * 1000)}us"

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


# ==============================================================================
# 3. Load + clean
# ==============================================================================
def load_rename_resample(filenames, col_map, freq, t_ref):
    """
    filenames : Liste de un ou plusieurs fichiers (shards) pour un même capteur
    col_map : Dictionnaire de renommage des colonnes
    freq : Fréquence native du capteur
    t_ref : Timestamp de référence absolue pour la synchronisation
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

        return resample_to_target_freq(df, freq, t_ref)

    except Exception as e:
        print(f"⚠ erreur sur les fichiers {filenames}: {e}")
        return None


# ==============================================================================
# 4. Breathing RPM
# ==============================================================================
def compute_breathing_rpm(path, t_ref):
    """
    Calcule la fréquence respiratoire (RPM) à partir du signal brut (chest_raw_breathing).
    Applique un filtre passe-bande et détecte les pics pour déduire la RPM.
    Ré-échantillonne le résultat final à 4Hz pour l'alignement avec les autres capteurs.
    """
    """
    Calcule la fréquence respiratoire (RPM) à partir du signal brut (chest_raw_breathing).
    Applique un filtre passe-bande et détecte les pics pour déduire la RPM.
    Ré-échantillonne le résultat final à la fréquence cible pour l'alignement avec les autres capteurs.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    ts = df["timestamp"].values
    sig = df["breathing_waveform"].values

    sig = sig - np.mean(sig)
    b, a = butter(2, [FMIN, FMAX], btype="bandpass", fs=FS_BREATHING)
    sig = filtfilt(b, a, sig)

    peaks, props = find_peaks(
        sig, distance=int(MIN_DIST_S * FS_BREATHING), prominence=PROM_MIN
    )

    rows = []
    for i in range(len(peaks) - 1):
        dt = (peaks[i + 1] - peaks[i]) / FS_BREATHING
        rpm = 60 / dt

        rows.append(
            {
                "timestamp": ts[peaks[i]],
                "breathing_rpm": rpm,
                "breathing_q": int(props["prominences"][i] >= PROM_OK),
            }
        )

    df = pd.DataFrame(rows)

    return resample_to_target_freq(df, 0.5, t_ref)


# ==============================================================================
# 5. Labels
# ==============================================================================
def assign_labels(df, intervals):
    """
    Assigne la colonne 'label' aux données resamplées en fonction des intervalles extraits.

    Comportement :
      - Zones labelisées (baseline, activity, fatigue) → label assigné normalement.
      - Zone entre 'activity' et 'fatigue' → label 'pre_fatigue' (classe 2).
      - Zone entre 'baseline' et 'activity' → label 'unlabeled' (classe -1).
      - Les données avant le premier intervalle ou après le dernier sont supprimées.
    """
    df = df.copy()
    df["label"] = None

    # 1. Assigner les labels des intervalles connus
    for s, e, l in intervals:
        mask = (df["timestamp"] >= s) & (df["timestamp"] <= e)
        df.loc[mask, "label"] = l

    # 2. Identifier les bornes globales de la session
    session_start = min(s for s, e, l in intervals)
    session_end   = max(e for s, e, l in intervals)

    # Supprimer les données hors session (avant t0 ou après la fin)
    df = df[(df["timestamp"] >= session_start) & (df["timestamp"] <= session_end)].copy()

    # 3. Identifier les intervalles activity et fatigue pour déduire la zone pre_fatigue
    activity_end = None
    fatigue_start = None
    for s, e, l in intervals:
        if l == "activity":
            activity_end = e
        if l == "fatigue":
            fatigue_start = s

    # 4. Zone entre activity et fatigue → pre_fatigue (classe 2)
    if activity_end is not None and fatigue_start is not None:
        mask_pre_fatigue = (
            (df["timestamp"] > activity_end)
            & (df["timestamp"] < fatigue_start)
            & (df["label"].isna())
        )
        df.loc[mask_pre_fatigue, "label"] = "pre_fatigue"
        print(f"  ℹ pre_fatigue assigné : {mask_pre_fatigue.sum()} lignes")

    # 5. Zone entre baseline et activity → unlabeled (classe -1)
    mask_unlabeled = df["label"].isna()
    df.loc[mask_unlabeled, "label"] = -1
    print(f"  ℹ unlabeled assigné  : {mask_unlabeled.sum()} lignes")

    print("  ℹ distribution labels :")
    print(df["label"].value_counts(dropna=False).to_string())

    return df


def data_clean_1(path):
    """
    Parcourt le répertoire brut et prépare le premier niveau de nettoyage :
      - Extraction des intervalles de label et de la référence t0.
      - Chargement et fusion des signaux (acc, eda, hr, ibi, temp) via merge_asof.
      - Ajout de la fréquence respiratoire.
      - Attribution des labels et filtrage des zones hors-label.

    Retourne : Un DataFrame pandas consolidé.
    """
    """
    Parcourt le répertoire brut et prépare le premier niveau de nettoyage :
      - Extraction des intervalles de label et de la référence t0.
      - Chargement et fusion des signaux (acc, eda, hr, ibi, temp) via merge_asof.
      - Ajout de la fréquence respiratoire.
      - Attribution des labels et filtrage des zones hors-label.

    Retourne : Un DataFrame pandas consolidé.
    """
    all_data = []

    for p_dir in path.iterdir():
        if not p_dir.is_dir():
            continue

        for s_dir in p_dir.iterdir():
            if not s_dir.is_dir():
                continue

            print(f"\n {p_dir.name} | {s_dir.name}")

            # 🔥 Extraction des intervalles et du t0 (référence temporelle absolue de la session)
            intervals, t0 = get_label_intervals(s_dir / PATH_MARKERS)
            if not intervals or t0 is None:
                continue

            merged = None

            # signals
            for f, cmap in FILE_COLS.items():
                # 🔥 Détection des shards (ex: wrist_acc.csv, wrist_acc_2.csv, etc.)
                base_name = f.replace(".csv", "")
                shards = list(s_dir.glob(f"{base_name}*.csv"))

                if not shards:
                    continue

                df = load_rename_resample(shards, cmap, SENSOR_FREQ[f], t0)
                if df is None:
                    continue

                merged = (
                    df
                    if merged is None
                    else pd.merge_asof(
                        merged.sort_values(COL_TIMESTAMP),
                        df.sort_values(COL_TIMESTAMP),
                        on=COL_TIMESTAMP,
                        tolerance=TARGET_PERIOD,
                        direction="nearest",
                    )
                )
            # breathing
            b_path = s_dir / PATH_BREATHING
            if b_path.exists():
                df_b = compute_breathing_rpm(b_path, t0)
                merged = (
                    df_b
                    if merged is None
                    else pd.merge_asof(
                        merged.sort_values(COL_TIMESTAMP),
                        df_b.sort_values(COL_TIMESTAMP),
                        on=COL_TIMESTAMP,
                        tolerance=TARGET_PERIOD,
                        direction="nearest",
                    )
                )

            if merged is None:
                continue

            # DEBUG
            print(
                "timestamp range:",
                merged[COL_TIMESTAMP].min(),
                merged[COL_TIMESTAMP].max(),
            )

            merged = assign_labels(merged, intervals)

            if merged.empty:
                print("toujours vide")
                continue

            merged[COL_PARTICIPANT] = p_dir.name
            merged[COL_SESSION] = s_dir.name

            all_data.append(merged)

    # ==============================================================================
    # SAVE
    # ==============================================================================

    if all_data:
        df = pd.concat(all_data)
        return df
    else:
        print("\n Toujours aucune donnée (vérifier timestamps)")



# ==============================================================================
# Variante SANS synchronisation de frequence (format B : union exacte des
# timestamps, NaN reels la ou un capteur n'a pas de mesure a cet instant)
# ==============================================================================

def load_rename_no_sync(filenames, col_map, t_ref):
    """
    Identique a load_rename_resample mais SANS resample_to_target_freq :
    conserve les timestamps natifs du capteur (pas d'interpolation/ffill de sync).
    """
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
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce") - t_ref
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df.reset_index(drop=True)


def compute_breathing_rpm_no_sync(path, t_ref):
    """
    Identique a compute_breathing_rpm mais SANS le resample_to_target_freq final :
    garde un point RPM par pic detecte, a son timestamp natif.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    ts = df["timestamp"].values
    sig = df["breathing_waveform"].values
    sig = sig - np.mean(sig)
    b, a = butter(2, [FMIN, FMAX], btype="bandpass", fs=FS_BREATHING)
    sig = filtfilt(b, a, sig)

    peaks, props = find_peaks(
        sig, distance=int(MIN_DIST_S * FS_BREATHING), prominence=PROM_MIN
    )

    rows = []
    for i in range(len(peaks) - 1):
        dt = (peaks[i + 1] - peaks[i]) / FS_BREATHING
        rpm = 60 / dt
        rows.append(
            {
                "timestamp": ts[peaks[i]] - t_ref,
                "breathing_rpm": rpm,
            }
        )

    return pd.DataFrame(rows)


def merge_sensors_outer(sensor_dfs):
    """
    Fusionne les DataFrames capteurs par UNION EXACTE des timestamps (merge outer).
    Contrairement a merge_asof (tolerance + nearest), aucune valeur n'est
    dupliquee/rapprochee : chaque capteur garde ses propres lignes, les autres
    colonnes restent NaN a cet instant si ce capteur n'a pas mesure a ce moment.
    """
    merged = None
    for df in sensor_dfs:
        if df is None or df.empty:
            continue
        merged = df if merged is None else pd.merge(merged, df, on="timestamp", how="outer")
    if merged is not None:
        merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def data_clean_1_no_sync(path):
    """
    Variante de data_clean_1 SANS synchronisation de frequence :
      - Meme extraction des intervalles de label et de t0
      - Chaque capteur garde ses timestamps natifs (pas de resample_to_target_freq)
      - Fusion par union exacte des timestamps (merge outer), pas merge_asof
      - Meme attribution des labels (assign_labels, inchangee)
    Retourne un DataFrame avec les memes colonnes que data_clean_1, mais avec
    des NaN reels la ou un capteur n'avait pas de mesure a ce timestamp precis
    (au lieu d'une valeur interpolee/dupliquee par la synchronisation 4Hz).
    """
    all_data = []

    for p_dir in path.iterdir():
        if not p_dir.is_dir():
            continue
        for s_dir in p_dir.iterdir():
            if not s_dir.is_dir():
                continue

            print(f"\n {p_dir.name} | {s_dir.name} (no_sync)")

            intervals, t0 = get_label_intervals(s_dir / PATH_MARKERS)
            if not intervals or t0 is None:
                continue

            sensor_dfs = []
            for f, cmap in FILE_COLS.items():
                base_name = f.replace(".csv", "")
                shards = list(s_dir.glob(f"{base_name}*.csv"))
                if not shards:
                    continue
                df_sensor = load_rename_no_sync(shards, cmap, t0)
                if df_sensor is not None and not df_sensor.empty:
                    sensor_dfs.append(df_sensor)

            b_path = s_dir / PATH_BREATHING
            if b_path.exists():
                df_b = compute_breathing_rpm_no_sync(b_path, t0)
                if df_b is not None and not df_b.empty:
                    sensor_dfs.append(df_b[["timestamp", "breathing_rpm"]])

            merged = merge_sensors_outer(sensor_dfs)
            if merged is None or merged.empty:
                print("  vide, session ignoree")
                continue

            print(
                "timestamp range:",
                merged["timestamp"].min(),
                merged["timestamp"].max(),
                f"| {len(merged)} lignes (union timestamps)",
            )

            merged = assign_labels(merged, intervals)
            if merged.empty:
                print("  toujours vide apres assign_labels")
                continue

            merged[COL_PARTICIPANT] = p_dir.name
            merged[COL_SESSION] = s_dir.name
            all_data.append(merged)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        print("\n Aucune donnee (no_sync).")
        return None


def clean_data_2_no_sync(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variante de clean_data_2 pour le pipeline SANS synchronisation de frequence.
    Ne fait PAS de ffill/bfill par groupe : cette propagation dupliquerait les
    valeurs d'un capteur lent (ex: ibi) sur toutes les lignes generees par les
    capteurs plus rapides entre deux vraies mesures — recreant artificiellement
    le meme probleme que la synchronisation de frequence qu'on cherche a eviter.
    Les NaN reels (capteur absent a ce timestamp precis) sont conserves tels quels.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 1 : NETTOYAGE (clean_data_2_no_sync, sans ffill/bfill)")
    print("=" * 60)

    df = df.copy()

    if "breathing_q" in df.columns:
        df = df.drop(columns=["breathing_q"])
        print("✔ breathing_q supprimé")

    nan_total = df[SIGNAL_COLS].isna().sum().sum() if all(c in df.columns for c in SIGNAL_COLS) else None
    print(f"ℹ NaN conservés tels quels (frequences non synchronisées) : {nan_total}")

    if "eda" in df.columns:
        skew_before = df["eda"].skew()
        df["eda"] = np.log1p(df["eda"])
        print(f"✔ EDA log1p | asymétrie : {skew_before:.3f} → {df['eda'].skew():.3f}")

    df = df.sort_values(
        by=[COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP], ascending=[True, True, True]
    ).reset_index(drop=True)
    print(f"✔ Trié par participant → session → timestamp")

    print(f"\n✔ clean_data_2_no_sync terminé | Shape : {df.shape}")
    return df# ==============================================================================
# Fusion des données démographiques (age, gender) depuis metadata.csv
# ==============================================================================


def merge_demographics(df: pd.DataFrame, metadata_path=None) -> pd.DataFrame:
    """
    Fusionne les colonnes 'age', 'gender' et les autres colonnes de questionnaires/métadonnées
    depuis metadata.csv dans le dataset.
      - Lecture de metadata.csv
      - Nettoyage des noms de colonnes et des valeurs textuelles
      - Mapping/encodage des variables qualitatives en valeurs numériques
      - Jointure sur la colonne 'participant' (zfill(2))
    Retourne : DataFrame avec toutes les colonnes fusionnées.
    """
    if metadata_path is None:
        metadata_path = METADATA_PATH

    meta = pd.read_csv(metadata_path)
    
    # Nettoyage des noms de colonnes (suppression des espaces blancs autour des noms)
    meta.columns = meta.columns.str.strip()

    # Création de la colonne participant avec padding (ex: 1 -> '01') pour correspondre aux dossiers
    meta[COL_PARTICIPANT] = meta["ID"].astype(str).str.zfill(2)

    # Encodage de gender (Female -> 0, Male -> 1)
    meta["gender"] = meta["gender"].astype(str).str.strip().str.lower().map({"female": 0, "male": 1})

    # Encodage des colonnes Likert (personnalité)
    likert_cols = [
        "reserved",
        "generally_trusting",
        "lazy",
        "relaxed_handless_stress",
        "few_artistic_interests",
        "sociable",
        "criticize_others",
        "thorough_job",
        "easily_nervous",
        "active_magination",
    ]
    likert_mapping = {
        "disagree strongly": 1,
        "disagree a little": 2,
        "neither agree nor disagree": 3,
        "agree a little": 4,
        "agree strongly": 5
    }
    for col in likert_cols:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip().str.lower().map(likert_mapping)

    # Encodage des colonnes de condition physique (fitness)
    fitness_cols = [
        "phisical_fitness",
        "cardiorespiratory_fitness",
        "muscular_strength",
        "agility_speed",
        "flexibility",
    ]
    fitness_mapping = {
        "very poor": 1,
        "poor": 2,
        "average": 3,
        "good": 4,
        "very good": 5
    }
    for col in fitness_cols:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip().str.lower().map(fitness_mapping)

    # Encodage des conditions de santé cardiovasculaire (No -> 0, Yes -> 1)
    yes_no_mapping = {"no": 0, "yes": 1}
    if "cardiovascular_health_conditions" in meta.columns:
        meta["cardiovascular_health_conditions"] = (
            meta["cardiovascular_health_conditions"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(yes_no_mapping)
        )

    # Sélection des colonnes à fusionner
    columns_to_merge = [
        COL_PARTICIPANT,
        "age",
        "gender",
        "reserved",
        "generally_trusting",
        "lazy",
        "relaxed_handless_stress",
        "few_artistic_interests",
        "sociable",
        "criticize_others",
        "thorough_job",
        "easily_nervous",
        "active_magination",
        "phisical_fitness",
        "cardiorespiratory_fitness",
        "muscular_strength",
        "agility_speed",
        "flexibility",
        "cardiovascular_health_conditions"
    ]
    
    # Filtrer les colonnes existantes au cas où certaines manqueraient dans le CSV
    columns_to_merge = [c for c in columns_to_merge if c in meta.columns]

    meta = meta[columns_to_merge]

    df = df.merge(meta, on=COL_PARTICIPANT, how="left")

    print(
        f"✔ Démographiques et métadonnées fusionnés | {len(columns_to_merge) - 1} colonnes ajoutées pour {meta.shape[0]} participants"
    )
    return df


# ==============================================================================
# visualition des données trouvé de data_clean_1
# ==============================================================================


def visualize_data(df, num_cols=None):
    """
    Affiche un résumé statistique et descriptif complet du dataset :
    types, valeurs manquantes, doublons, distribution des labels,
    statistiques par session/participant, continuité temporelle et corrélations.

    num_cols : liste des colonnes numeriques a analyser (defaut : NUM_COLS de config.py).
               Utile pour le mode no_sync ou breathing_q n'existe pas.
    """
    if num_cols is None:
        num_cols = NUM_COLS

    # ==============================================================================
    # Noms réels des colonnes
    # ==============================================================================

    print("=" * 70)
    print("1. INFORMATIONS GÉNÉRALES")
    print("=" * 70)
    print(f"Lignes                : {len(df)}")
    print(f"Colonnes              : {len(df.columns)}")
    print(
        f"Taille mémoire        : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )
    print(f"\nColonnes              : {df.columns.tolist()}")

    print("\n" + "=" * 70)
    print("2. TYPES DES COLONNES")
    print("=" * 70)
    print(df.dtypes.to_string())

    print("\n" + "=" * 70)
    print("3. APERÇU DES DONNÉES")
    print("=" * 70)
    print("\n--- Premières lignes ---")
    print(df.head(5).to_string())
    print("\n--- Dernières lignes ---")
    print(df.tail(5).to_string())

    print("\n" + "=" * 70)
    print("4. VALEURS MANQUANTES")
    print("=" * 70)
    missing = df.isna().sum()
    missing_pct = (df.isna().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Nb manquants": missing, "Pourcentage": missing_pct})
    missing_df = missing_df[missing_df["Nb manquants"] > 0].sort_values(
        "Pourcentage", ascending=False
    )
    if missing_df.empty:
        print("✔ Aucune valeur manquante")
    else:
        print(missing_df.to_string())
    print(f"\nTotal valeurs manquantes   : {df.isna().sum().sum()}")
    print(f"Lignes avec au moins 1 NaN : {df.isna().any(axis=1).sum()}")

    print("\n" + "=" * 70)
    print("5. DOUBLONS")
    print("=" * 70)
    n_duplicates = df.duplicated().sum()
    print(f"Lignes entièrement dupliquées : {n_duplicates}")
    print(f"Pourcentage                   : {n_duplicates / len(df) * 100:.2f}%")

    n_dup_ts = df.duplicated(subset=[COL_TIMESTAMP]).sum()
    print(f"\nTimestamps dupliqués          : {n_dup_ts}")
    print(f"Pourcentage                   : {n_dup_ts / len(df) * 100:.2f}%")

    print(f"\n--- Analyse des timestamps dupliqués ---")
    dup_ts = df[df.duplicated(subset=[COL_TIMESTAMP], keep=False)]
    print(f"Participants concernés : {dup_ts[COL_PARTICIPANT].unique()}")
    print(f"Sessions concernées    : {dup_ts[COL_SESSION].unique()}")
    print(f"\nExemple de timestamps dupliqués :")
    print(
        df[df.duplicated(subset=[COL_TIMESTAMP], keep=False)][
            [COL_TIMESTAMP, COL_PARTICIPANT, COL_SESSION, COL_LABEL]
        ]
        .head(10)
        .to_string()
    )

    print(f"\nDoublons par participant/session :")
    dup_by_group = (
        df.groupby([COL_PARTICIPANT, COL_SESSION])
        .apply(lambda x: x.duplicated().sum(), include_groups=False)
        .reset_index()
    )
    dup_by_group.columns = [COL_PARTICIPANT, COL_SESSION, "nb_doublons"]
    result = dup_by_group[dup_by_group["nb_doublons"] > 0]
    print(result.to_string() if not result.empty else "✔ Aucun doublon par groupe")

    print("\n" + "=" * 70)
    print("6. STATISTIQUES DESCRIPTIVES")
    print("=" * 70)
    stats = df[num_cols].describe().T
    stats["median"] = df[num_cols].median()
    stats["skewness"] = df[num_cols].skew()
    stats["kurtosis"] = df[num_cols].kurt()
    print(
        stats[
            [
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "median",
                "75%",
                "max",
                "skewness",
                "kurtosis",
            ]
        ]
        .round(3)
        .to_string()
    )

    print("\n" + "=" * 70)
    print("7. DISTRIBUTION DES LABELS")
    print("=" * 70)
    label_counts = df[COL_LABEL].value_counts()
    label_pct = df[COL_LABEL].value_counts(normalize=True) * 100
    label_df = pd.DataFrame(
        {"Nb lignes": label_counts, "Pourcentage": label_pct.round(2)}
    )
    print(label_df.to_string())
    print(
        f"\nClasses équilibrées : "
        f"{'✔ Oui' if label_pct.max() - label_pct.min() < 10 else '✗ Non (déséquilibré)'}"
    )

    print("\n" + "=" * 70)
    print("8. DISTRIBUTION PAR PARTICIPANT")
    print("=" * 70)
    part_stats = (
        df.groupby(COL_PARTICIPANT)
        .agg(
            nb_lignes=(COL_TIMESTAMP, "count"),
            nb_sessions=(COL_SESSION, "nunique"),
        )
        .reset_index()
    )
    print(part_stats.to_string())

    print("\n" + "=" * 70)
    print("9. DISTRIBUTION PAR SESSION")
    print("=" * 70)
    sess_stats = (
        df.groupby([COL_PARTICIPANT, COL_SESSION])
        .agg(
            nb_lignes=(COL_TIMESTAMP, "count"),
            label_dist=(COL_LABEL, lambda x: x.value_counts().to_dict()),
        )
        .reset_index()
    )
    print(sess_stats.to_string())

    print("\n" + "=" * 70)
    print("10. CONTINUITÉ TEMPORELLE")
    print("=" * 70)
    for (pid, sid), group in df.groupby([COL_PARTICIPANT, COL_SESSION]):
        group = group.sort_values(COL_TIMESTAMP)
        diffs = group[COL_TIMESTAMP].diff().dropna()
        diffs = diffs[diffs > 0]

        expected_interval = TARGET_PERIOD
        anomalies = diffs[diffs > expected_interval * 2]

        print(
            f"  P{pid}/S{sid} → "
            f"intervalle moy={diffs.mean():.1f}ms | "
            f"min={diffs.min():.1f}ms | "
            f"max={diffs.max():.1f}ms | "
            f"sauts>{expected_interval * 2}ms : {len(anomalies)}"
        )

    print("\n" + "=" * 70)
    print("11. VALEURS ABERRANTES (outliers IQR)")
    print("=" * 70)
    print(f"{'Colonne':<20} {'Outliers':<10} {'%':<8} {'Min':<12} {'Max':<12}")
    print("-" * 65)
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        pct = len(outliers) / len(df) * 100
        print(
            f"{col:<20} {len(outliers):<10} {pct:<8.2f} "
            f"{df[col].min():<12.3f} {df[col].max():<12.3f}"
        )

    print("\n" + "=" * 70)
    print("12. CORRÉLATIONS ENTRE SIGNAUX")
    print("=" * 70)
    corr_matrix = df[num_cols].corr().round(3)
    print(corr_matrix.to_string())

    print("\n" + "=" * 70)
    print("13. STATISTIQUES PAR LABEL")
    print("=" * 70)
    for lbl in df[COL_LABEL].unique():
        if lbl is None or (isinstance(lbl, float) and pd.isna(lbl)):
            print(f"\n--- Label : NaN (unlabeled - zone entre baseline et activity) ---")
            subset = df[df[COL_LABEL].isna()][num_cols]
        else:
            print(f"\n--- Label : {str(lbl).upper()} ---")
            subset = df[df[COL_LABEL] == lbl][num_cols]
        print(subset.describe().T[["mean", "std", "min", "max"]].round(3).to_string())

    print("\n" + "=" * 70)
    print("14. VÉRIFICATION COHÉRENCE TEMPORELLE")
    print("=" * 70)
    ts_min = df[COL_TIMESTAMP].min()
    ts_max = df[COL_TIMESTAMP].max()
    print(f"Timestamp min : {ts_min:.0f} ({pd.to_datetime(ts_min, unit='ms')})")
    print(f"Timestamp max : {ts_max:.0f} ({pd.to_datetime(ts_max, unit='ms')})")
    print(f"Durée totale  : {(ts_max - ts_min) / 1000 / 3600:.2f} heures")

    print("\n" + "=" * 70)
    print("15. RÉSUMÉ FINAL")
    print("=" * 70)
    print(f"✔ Lignes totales           : {len(df)}")
    print(f"✔ Participants             : {df[COL_PARTICIPANT].nunique()}")
    print(f"✔ Sessions par participant : {df[COL_SESSION].nunique()}")
    print(f"✔ Colonnes signaux         : {len(num_cols)}")
    print(
        f"{'✔' if n_duplicates == 0 else '✗'} Doublons                  : {n_duplicates}"
    )
    print(
        f"{'✔' if n_dup_ts == 0 else '⚠'} Timestamps dupliqués      : {n_dup_ts} ({n_dup_ts / len(df) * 100:.1f}%)"
    )
    print(
        f"{'✔' if df.isna().sum().sum() == 0 else '✗'} Valeurs manquantes        : {df.isna().sum().sum()}"
    )
    print(
        f"{'✔' if label_pct.max() - label_pct.min() < 10 else '⚠'} Équilibre des classes     : {label_df['Pourcentage'].to_dict()}"
    )

# ──  Supprime breathing_q ,ffill/bfill par groupe  pour traite les NaN de ( ibi, temp, wrist_hr, eda) puis median ,   et Log transform EDA  log(1+x) (réduire asymétrie ) ,─────────────────────────────────────────
def clean_data_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le dataset brut :
      - Supprime breathing_q (signal constant inutile)
      - Traite les NaN par ffill/bfill par groupe participant/session
      - Fallback médiane globale si NaN restants
      - Log transform sur EDA pour réduire l'asymétrie
      - Trie par participant puis session
    Retourne : df nettoyé (float64)
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 1 : NETTOYAGE (clean_data)")
    print("=" * 60)

    df = df.copy()

    # 1.1 ── Supprimer breathing_q ────────────────────────────────────────
    if "breathing_q" in df.columns:
        df = df.drop(columns=["breathing_q"])
        print("✔ breathing_q supprimé")

    # 1.2 ── NaN par ffill/bfill par groupe ───────────────────────────────
    nan_before = df[SIGNAL_COLS].isna().sum().sum()
    print(f"\nNaN avant : {nan_before}")

    df[SIGNAL_COLS] = df.groupby([COL_PARTICIPANT, COL_SESSION])[SIGNAL_COLS].transform(
        lambda x: x.ffill().bfill()
    )

    nan_after = df[SIGNAL_COLS].isna().sum().sum()

    # Fallback médiane si NaN restants (groupe avec une seule valeur NaN)
    if nan_after > 0:
        print(f"⚠ {nan_after} NaN restants → médiane globale")
        for col in SIGNAL_COLS:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

    print(f"✔ NaN traités : {nan_before} → {df[SIGNAL_COLS].isna().sum().sum()}")

    # 1.3 ── Log transform EDA ────────────────────────────────────────────
    if "eda" in df.columns:
        skew_before = df["eda"].skew()
        df["eda"] = np.log1p(df["eda"])
        print(f"✔ EDA log1p | asymétrie : {skew_before:.3f} → {df['eda'].skew():.3f}")

    # 1.4 ── Tri participant / session ────────────────────────────────────
    df = df.sort_values(
        by=[COL_PARTICIPANT, COL_SESSION], ascending=[True, True]
    ).reset_index(drop=True)
    print(f"✔ Trié par participant → session")

    print(f"\n✔ clean_data terminé | Shape : {df.shape}")
    return df


# ==============================================================================
# Suppression des doublons
# ==============================================================================


def remove_duplicates(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """
    Supprime les doublons du dataset.

    Stratégie :
      1. Doublons exacts (toutes colonnes)  -> supprimes en premier
      2. Doublons de timestamp par groupe (participant, session) -> garder keep

    Parametres
    ----------
    df   : DataFrame a nettoyer
    keep : 'first' (defaut) | 'last' | False (supprime toutes les occurrences)

    Retourne
    --------
    DataFrame nettoye.
    """
    print("\n" + "=" * 60)
    print("SUPPRESSION DES DOUBLONS (remove_duplicates)")
    print("=" * 60)

    df = df.copy()
    n0 = len(df)

    # 1. Doublons exacts
    df = df.drop_duplicates(keep=keep)
    n1 = len(df)
    print(f"Doublons exacts supprimes           : {n0 - n1} ({n0} -> {n1})")

    # 2. Doublons de timestamp par groupe participant/session
    key_cols = [
        c for c in [COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP] if c in df.columns
    ]
    df = df.drop_duplicates(subset=key_cols, keep=keep).reset_index(drop=True)
    n2 = len(df)
    print(f"Doublons timestamp/groupe supprimes : {n1 - n2} ({n1} -> {n2})")

    print(f"\nremove_duplicates termine | Shape : {df.shape}")
    return df


# ── Label encodé : baseline=0 | activity=1 | pre_fatigue=2 | fatigue=3  et Séparation X / y─────────────────────
def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Encode le label et sépare features / target :
      - baseline → 0  |  activity → 1  |  pre_fatigue → 2  |  fatigue → 3
      - La zone baseline↔activity reste à -1 pour le semi-supervisé.
      - Encodage dans la colonne 'label' (pas de colonne séparée)
      - Conversion float32 pour compatibilité STM32H7 (FPU 32-bit)
    Retourne : (X, y)
      X → DataFrame des signaux (float32)
      y → Series du label encodé (int8)
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : ENCODAGE (encode_features)")
    print("=" * 60)

    df = df.copy()

    # 2.1 ── Écarter seulement les vrais labels manquants ──────────────
    n_unlabeled = df[COL_LABEL].isna().sum()
    if n_unlabeled > 0:
        print(f"⚠ {n_unlabeled} lignes avec label NaN écartées de X/y")
    df = df.dropna(subset=[COL_LABEL]).copy()

    # 2.2 ── Encoder le label ────────────────────────────────────────────────────────────────────────────────────
    known_labels = set(LABEL_MAP.keys()) | {-1}
    unknown = set(df[COL_LABEL].unique()) - known_labels
    if unknown:
        print(f"⚠ Labels inconnus (seront écartés) : {unknown}")

    df[COL_LABEL] = df[COL_LABEL].replace(LABEL_MAP)

    # Écarter les lignes dont le label n’était pas dans LABEL_MAP ou -1
    valid_encoded_labels = set(LABEL_MAP.values()) | {-1}
    mask_unknown = ~df[COL_LABEL].isin(valid_encoded_labels)
    n_unknown = mask_unknown.sum()
    if n_unknown > 0:
        print(f"⚠ {n_unknown} lignes avec label inconnu écartées après mapping")
        df = df.loc[~mask_unknown].copy()

    print(f"✔ Label encodé : baseline=0 | activity=1 | pre_fatigue=2 | fatigue=3")
    print(f"\n   Distribution :")
    print(df[COL_LABEL].value_counts().sort_index().to_string())

    # 2.3 ── Séparation X / y ───────────────────────────────────────────────────────────────────────────────────
    X = df[SIGNAL_COLS].astype(np.float32)  # float32 → STM32H7 FPU natif
    y = df[COL_LABEL].astype(np.int8)  # int8    → économie mémoire

    print(f"\n✔ X shape : {X.shape} | dtype : {X.dtypes.unique()}")
    print(f"✔ y shape : {y.shape} | dtype : {y.dtype}")
    print(f"\n✔ encode_features terminé")
    return X, y


# ==============================================================================
# RobustScaler (robuste aux outliers, Q1/Q3)
# ==============================================================================


def normalize_features(
    X: pd.DataFrame, scaler: RobustScaler = None, fit: bool = True
) -> tuple[pd.DataFrame, RobustScaler]:
    """
    Normalise les features avec RobustScaler (robuste aux outliers, Q1/Q3) :
      - fit=True  → calcule le scaler sur X (à utiliser sur le train set)
      - fit=False → applique un scaler existant  (à utiliser sur le test set)
    Retourne : (X_normalized, scaler)
    ⚠ En LOSO : toujours fit=True sur le train, fit=False sur le test.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : NORMALISATION (normalize_features)")
    print("=" * 60)

    X = X.copy()

    if fit:
        scaler = RobustScaler()
        X[SIGNAL_COLS] = scaler.fit_transform(X[SIGNAL_COLS])
        print(f"✔ RobustScaler fitted et appliqué")
    else:
        if scaler is None:
            raise ValueError(
                "Fournir un scaler quand fit=False (ex: fold test en LOSO)"
            )
        X[SIGNAL_COLS] = scaler.transform(X[SIGNAL_COLS])
        print(f"✔ RobustScaler appliqué (pré-calculé)")

    print(f"\n   Stats après normalisation :")
    print(X[SIGNAL_COLS].agg(["mean", "std"]).round(3).to_string())
    print(f"\n✔ normalize_features terminé | Shape : {X.shape}")
    return X, scaler


# ==============================================================================
# Conversion float64 → float32
# ==============================================================================


def reduce_precision(df):
    """
    Convertir les colonnes signal de float64 → float32
    - Divise la RAM par 2
    - Compatible FPU STM32H7 (32-bit float natif)
    - Précision suffisante pour les signaux physiologiques
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : RÉDUCTION PRÉCISION (reduce_precision)")
    print("=" * 60)

    df = df.copy()

    ram_before = df[SIGNAL_COLS].memory_usage(deep=True).sum() / 1024

    df[SIGNAL_COLS] = df[SIGNAL_COLS].astype(np.float32)

    ram_after = df[SIGNAL_COLS].memory_usage(deep=True).sum() / 1024

    print(f"✔ float64 → float32")
    print(f"   RAM avant  : {ram_before:.1f} KB")
    print(f"   RAM après  : {ram_after:.1f} KB")
    print(f"   Gain       : {ram_before - ram_after:.1f} KB ({50}% économisé)")
    print(f"   Précision  : ~15 chiffres → ~7 chiffres (suffisant)")

    # Vérification visuelle
    print(f"\n   Exemple ibi avant : 1.64285714285714...")
    print(f"   Exemple ibi après : {df['ibi'].iloc[0]:.7f}")

    return df
