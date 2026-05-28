# --------------------------------------------------------------------// Importations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import RobustScaler
try:
    from .config import *
except ImportError:
    from config import *

# --------------------------------------------------------------------// Fonctions de préparation des données

PATH_BREATHING = "chest_raw_breathing.csv"
PATH_MARKERS = "exp_markers.csv"

# ==============================================================================
# Paramètres respiration
# ==============================================================================
FS_BREATHING = 25.0
FMIN, FMAX   = 0.05, 0.80
MIN_DIST_S   = 1.5
PROM_MIN     = 100
PROM_OK      = 500

# ==============================================================================
# Fréquences capteurs
# ==============================================================================
SENSOR_FREQ = {
    "chest_physiology_summary.csv": 1,
    "wrist_acc.csv":                32,
    "wrist_eda.csv":                4,
    "wrist_hr.csv":                 1,
    "wrist_ibi.csv":                0.59,
    "wrist_skin_temperature.csv":   4,
}

TARGET_FREQ   = 1
TARGET_PERIOD = 1000 / TARGET_FREQ  # ms

# ==============================================================================
# Colonnes
# ==============================================================================
FILE_COLS = {
    "wrist_acc.csv": {"ax": "acc_x", "ay": "acc_y", "az": "acc_z"},
    "wrist_eda.csv": {"eda": "eda"},
    "wrist_hr.csv": {"hr": "wrist_hr"},
    "wrist_ibi.csv": {"duration": "ibi"},
    "wrist_skin_temperature.csv": {"temp": "temp"},
}

LABELS_OF_INTEREST = ["baseline", "activity", "fatigue"]

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
    b,a = butter(2, [FMIN, FMAX], btype="bandpass", fs=FS_BREATHING)
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

    return resample_to_target_freq(df, 0.5, t_ref)



# ==============================================================================
# 5. Labels
# ==============================================================================
def assign_labels(df, intervals):
    """
    Assigne la colonne 'label' aux données resamplées en fonction des intervalles extraits.
    Supprime les lignes ne correspondant à aucun label d'intérêt.
    """
    """
    Assigne la colonne 'label' aux données resamplées en fonction des intervalles extraits.
    Supprime les lignes ne correspondant à aucun label d'intérêt.
    """
    df["label"] = None

    for s, e, l in intervals:
        mask = (df["timestamp"] >= s) & (df["timestamp"] <= e)
        df.loc[mask, "label"] = l

    print("  ℹ labels trouvés:", df["label"].value_counts(dropna=False))

    return df.dropna(subset=["label"])


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
# Fusion des données démographiques (age, gender) depuis metadata.csv
# ==============================================================================

def merge_demographics(df: pd.DataFrame, metadata_path=None) -> pd.DataFrame:
    """
    Fusionne les colonnes 'age' et 'gender' depuis metadata.csv dans le dataset.
      - Lecture de metadata.csv (colonnes: participant_id, age, gender, ...)
      - Jointure sur la colonne 'participant'
      - Mapping : 'Female' -> 0, 'Male' -> 1
    Retourne : DataFrame avec colonnes 'age' (int) et 'gender' (0/1) ajoutées.
    """
    meta = pd.read_csv(metadata_path, dtype={"participant_id": str})

    # Mapping Female -> 0, Male -> 1
    meta["gender"] = meta["gender"].map({"Female": 0, "Male": 1})

    meta = meta.rename(columns={"participant_id": COL_PARTICIPANT})[
        [COL_PARTICIPANT, "age", "gender"]
    ]

    df = df.merge(meta, on=COL_PARTICIPANT, how="left")

    print(f"✔ Démographiques fusionnés | age/gender (encoded) ajoutés pour {meta.shape[0]} participants")
    return df



# ==============================================================================
# visualition des données trouvé de data_clean_1
# ==============================================================================

def visualize_data(df):
    """
    Affiche un résumé statistique et descriptif complet du dataset :
    types, valeurs manquantes, doublons, distribution des labels,
    statistiques par session/participant, continuité temporelle et corrélations.
    """
    """
    Affiche un résumé statistique et descriptif complet du dataset :
    types, valeurs manquantes, doublons, distribution des labels,
    statistiques par session/participant, continuité temporelle et corrélations.
    """


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

    # Doublons sur timestamp uniquement
    n_dup_ts = df.duplicated(subset=[COL_TIMESTAMP]).sum()
    print(f"\nTimestamps dupliqués          : {n_dup_ts}")
    print(f"Pourcentage                   : {n_dup_ts / len(df) * 100:.2f}%")


# ==============================================================================
# Investiguer les timestamps dupliqués
# ==============================================================================

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

    # Doublons par participant/session
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
    stats = df[NUM_COLS].describe().T
    stats["median"] = df[NUM_COLS].median()
    stats["skewness"] = df[NUM_COLS].skew()
    stats["kurtosis"] = df[NUM_COLS].kurt()
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
        diffs = diffs[diffs > 0]  # ignorer les timestamps identiques

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
    for col in NUM_COLS:
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
    corr_matrix = df[NUM_COLS].corr().round(3)
    print(corr_matrix.to_string())

    print("\n" + "=" * 70)
    print("13. STATISTIQUES PAR LABEL")
    print("=" * 70)
    for lbl in df[COL_LABEL].unique():
        print(f"\n--- Label : {lbl.upper()} ---")
        subset = df[df[COL_LABEL] == lbl][NUM_COLS]
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
    print(f"✔ Colonnes signaux         : {len(NUM_COLS)}")
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


# ── Label encodé : baseline=-1 | activity=0 | fatigue=1  et Séparation X / y─────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Encode le label et sépare features / target :
      - baseline → -1  |  activity → 0  |  fatigue → 1
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

    # 2.1 ── Encoder le label ─────────────────────────────────────────────
    unknown = set(df[COL_LABEL].unique()) - set(LABEL_MAP.keys())
    if unknown:
        print(f"⚠ Labels inconnus : {unknown}")

    df[COL_LABEL] = df[COL_LABEL].map(LABEL_MAP)

    print(f"✔ Label encodé : baseline=-1 | activity=0 | fatigue=1")
    print(f"\n   Distribution :")
    print(df[COL_LABEL].value_counts().sort_index().to_string())

    # 2.2 ── Séparation X / y ─────────────────────────────────────────────
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
