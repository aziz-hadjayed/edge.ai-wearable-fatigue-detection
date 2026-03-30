## `src/config.py` — Configuration centrale du projet

Ce fichier centralise tous les paramètres globaux du projet : chemins, colonnes, mappings et hyperparamètres. Il est importé par les autres modules pour éviter toute duplication de constantes.

---

### Chemins du projet

| Variable | Valeur résolue | Rôle |
|---|---|---|
| `BASE_DIR` | racine du projet | Chemin absolu vers la racine, déduit de l'emplacement de ce fichier |
| `DATA_DIR` | `BASE_DIR/data` | Dossier principal des données |
| `RAW_DIR` | `data/01_raw` | Données brutes originales (ne pas modifier) |
| `INTERIM_DIR` | `data/02_interim` | Données intermédiaires en cours de traitement |
| `PROCESSED_DIR` | `data/03_processed` | Données finales prêtes pour l'entraînement |
| `DATA_PROCESSED` | `03_processed/dataset_final.csv` | Fichier CSV consolidé final |
| `OUTPUT_PATH` | identique à `DATA_PROCESSED` | Alias pour le chemin de sortie |
| `MODELS_DIR` | `BASE_DIR/models_saved` | Dossier de sauvegarde des modèles entraînés |

---

### Paramètres de split train/test

```python
TEST_SIZE    = 0.2   # 20 % des données réservés au test
RANDOM_STATE = 42    # Graine pour la reproductibilité
```

---

### Colonnes clés du dataset

```python
COL_PARTICIPANT = "participant"   # Identifiant du sujet
COL_SESSION     = "session"       # Numéro de session
COL_TIMESTAMP   = "timestamp"     # Horodatage de la mesure
COL_LABEL       = "label"         # Étiquette cible (baseline / activity / fatigue)
```

---

### Colonnes numériques pour la visualisation (`NUM_COLS`)

```python
NUM_COLS = [
    "acc_x", "acc_y", "acc_z",     # Accélération (3 axes)
    "eda",                          # Activité électrodermale
    "wrist_hr",                     # Fréquence cardiaque (poignet)
    "ibi",                          # Intervalle inter-battements
    "temp",                         # Température cutanée
    "breathing_rpm",                # Fréquence respiratoire
    "breathing_q"                   # Qualité du signal respiratoire
]
```

---

### Colonnes de signaux pour le nettoyage et la normalisation (`SIGNAL_COLS`)

Identiques à `NUM_COLS` sauf `breathing_q` (exclue du pipeline de traitement) :

```python
SIGNAL_COLS = [
    "acc_x", "acc_y", "acc_z",
    "eda", "wrist_hr", "ibi",
    "temp", "breathing_rpm"
]
```

---

### Encodage des labels (`LABEL_MAP`)

```python
LABEL_MAP = {
    "baseline":  -1,   # État de repos de référence
    "activity":   0,   # Activité physique normale
    "fatigue":    1    # État de fatigue (classe cible)
}
```

---

### Hyperparamètres des modèles (`MODEL_PARAMS`)

```python
MODEL_PARAMS = {
    "RandomForest": {"n_estimators": 100, "max_depth": 10},
    "SVM":          {"kernel": "rbf", "C": 1.0},
}
```

| Modèle | Paramètre | Valeur | Description |
|---|---|---|---|
| RandomForest | `n_estimators` | 100 | Nombre d'arbres de décision |
| RandomForest | `max_depth` | 10 | Profondeur maximale de chaque arbre |
| SVM | `kernel` | `"rbf"` | Noyau à base radiale (Radial Basis Function) |
| SVM | `C` | 1.0 | Paramètre de régularisation |

---

## `src/data_prep.py` — Pipeline de préparation des données

Ce fichier orchestre les 4 étapes du pipeline de préparation : chargement/fusion des capteurs, nettoyage, encodage et normalisation. Il s'appuie sur `config.py` et `fonctions_principal_clean_data_1.py`.

---

### Vue d'ensemble du pipeline

```
RAW data (01_raw/)
      │
      ▼
 data_clean_1()      ← fusion multi-capteurs + labels + breathing_rpm
      │
      ▼
 visualize_data()    ← audit qualité (optionnel, console)
      │
      ▼
 clean_data_2()      ← NaN, log EDA, tri
      │
      ▼
 encode_features()   ← label → int, X/y split, float32
      │
      ▼
normalize_features() ← RobustScaler (Q1/Q3)
      │
      ▼
 reduce_precision()  ← float64 → float32
      │
      ▼
dataset_final.csv (03_processed/)
```

---

### `data_clean_1(path)` — Fusion multi-capteurs & labellisation

Parcourt l'arborescence `participant/ → session/` et pour chaque session :

1. Lit les intervalles de labels depuis le fichier markers (`get_label_intervals`)
2. Charge et renomme chaque fichier capteur, rééchantillonne à **4 Hz** (`load_rename_resample`)
3. Fusionne les signaux par `merge_asof` sur `timestamp` avec une tolérance de `TARGET_PERIOD`
4. Calcule `breathing_rpm` par filtrage Butterworth passe-bande (`compute_breathing_rpm`)
5. Assigne les labels (`assign_labels`) et ajoute les colonnes `participant` / `session`

**Retourne :** un DataFrame concaténé de toutes les sessions.

| Paramètre | Type | Description |
|---|---|---|
| `path` | `Path` | Chemin vers `data/01_raw/` |

---

### `visualize_data(df)` — Audit qualité du dataset brut

Affiche 15 sections de diagnostic en console :

| Section | Contenu |
|---|---|
| 1 | Dimensions, mémoire, liste des colonnes |
| 2 | Types des colonnes |
| 3 | Aperçu head/tail |
| 4 | Valeurs manquantes (nb + %) |
| 5 | Doublons globaux et par timestamp |
| 6 | Statistiques descriptives (mean, std, skewness, kurtosis) |
| 7 | Distribution des labels + équilibre des classes |
| 8 | Distribution par participant |
| 9 | Distribution par session |
| 10 | Continuité temporelle (intervalle moyen, sauts > 500 ms) |
| 11 | Outliers IQR par colonne |
| 12 | Matrice de corrélation |
| 13 | Statistiques par label |
| 14 | Cohérence timestamps (min/max/durée) |
| 15 | Résumé final avec indicateurs ✔/✗/⚠ |

---

### `clean_data_2(df)` — Nettoyage du dataset

| Étape | Action | Détail |
|---|---|---|
| 1.1 | Suppression `breathing_q` | Signal constant, inutile pour la modélisation |
| 1.2 | Imputation NaN | `ffill` puis `bfill` par groupe `participant/session` ; médiane globale en fallback |
| 1.3 | Log transform EDA | `log1p(eda)` pour réduire l'asymétrie de distribution |
| 1.4 | Tri | Par `participant` puis `session` (ordre croissant) |

**Retourne :** `pd.DataFrame` nettoyé en `float64`.

---

### `encode_features(df)` — Encodage & séparation X / y

| Étape | Action | Détail |
|---|---|---|
| 2.1 | Encodage label | `LABEL_MAP` → baseline=-1, activity=0, fatigue=1 |
| 2.2 | Séparation | `X = SIGNAL_COLS` en **float32** ; `y = label` en **int8** |

> Le choix **float32** est motivé par la cible embarquée STM32H7 dont la FPU est nativement 32-bit.

**Retourne :** `(X: pd.DataFrame, y: pd.Series)`

---

### `normalize_features(X, scaler, fit)` — Normalisation RobustScaler

Utilise `RobustScaler` (centrage sur la médiane, mise à l'échelle Q1/Q3), robuste aux outliers physiologiques.

| Paramètre | Type | Valeur par défaut | Description |
|---|---|---|---|
| `X` | `pd.DataFrame` | — | Features à normaliser |
| `scaler` | `RobustScaler` | `None` | Scaler pré-calculé (requis si `fit=False`) |
| `fit` | `bool` | `True` | `True` → fit+transform (train) ; `False` → transform seul (test/LOSO) |

> **Règle LOSO :** toujours `fit=True` sur le fold train, `fit=False` sur le fold test pour éviter toute fuite de données.

**Retourne :** `(X_normalized: pd.DataFrame, scaler: RobustScaler)`

---

### `reduce_precision(df)` — Réduction float64 → float32

Convertit les colonnes `SIGNAL_COLS` de `float64` à `float32`.

| Avantage | Détail |
|---|---|
| RAM divisée par 2 | `float64` = 8 octets → `float32` = 4 octets par valeur |
| Compatible STM32H7 | FPU 32-bit natif, pas de conversion au déploiement |
| Précision suffisante | ~7 chiffres significatifs, largement suffisant pour des signaux physiologiques |

**Retourne :** `pd.DataFrame` avec colonnes signaux en `float32`.
