import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Charger les données
file_path = "/media/mohamedaziz-hadjayed/D/aziz_data/fatigue_detection/edge-ai-wearable-fatigue-detection/data/01_raw/pre_task_survey.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 2. Identifier les 8 colonnes numériques
colonnes_8 = [
    "How alert do you feel?",
    "How sad do you feel?",
    "How tense do you feel?",
    "How much of an effort is it to do anything?",
    "How happy do you feel?",
    "How weary do you feel?",
    "How calm do you feel?",
    "How sleepy do you feel?"
]

# Vérifier qu'elles existent
missing_cols = [col for col in colonnes_8 if col not in df.columns]
if missing_cols:
    raise KeyError(f"Colonnes manquantes : {missing_cols}")

X = df[colonnes_8].values

# 3. Supprimer les lignes avec des NaN (si nécessaire)
X = X[~np.isnan(X).any(axis=1)]

# 4. Standardisation (centrer-réduire)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. PCA
pca = PCA()
pca.fit(X_scaled)

# 6. Variance expliquée
explained_variance = pca.explained_variance_ratio_
cumsum_variance = np.cumsum(explained_variance)

print("Variance expliquée par composante :")
for i, var in enumerate(explained_variance[:4], 1):
    print(f"PC{i} : {var:.2%}")
print(f"Cumulée après 4 composantes : {cumsum_variance[3]:.2%}")

# 7. Trouver les 4 colonnes originales les plus contributives sur les 4 premières PCs
# Méthode : somme des cos² (carré des loadings) sur les 4 premières PCs
loadings = pca.components_[:4, :]  # 4 PCs x 8 variables
cos2 = loadings ** 2
total_cos2 = cos2.sum(axis=0)      # somme sur les 4 PCs

# Associer aux noms de colonnes
cos2_df = pd.DataFrame({
    "Variable": colonnes_8,
    "Contribution (cos² total sur 4 PCs)": total_cos2
}).sort_values("Contribution (cos² total sur 4 PCs)", ascending=False)

print("\nColonnes classées par capacité à représenter l'espace PCA (4 dimensions) :")
print(cos2_df)

# 8. Sélectionner les 4 meilleures
top4_vars = cos2_df["Variable"].iloc[:4].tolist()
print("\n➡️ Les 4 colonnes à garder :", top4_vars)

# Variante optionnelle : une variable par composante principale
print("\n--- Variante : 1 variable par PC ---")
selected_vars = []
for i in range(4):
    idx_max = np.argmax(np.abs(loadings[i]))
    var_max = colonnes_8[idx_max]
    if var_max not in selected_vars:
        selected_vars.append(var_max)
    if len(selected_vars) == 4:
        break

print("➡️ 4 colonnes (1 par PC) :", selected_vars)