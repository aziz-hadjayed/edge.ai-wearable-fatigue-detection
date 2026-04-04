import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration des chemins
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    from config import DATA_PROCESSED, COL_PARTICIPANT, SIGNAL_COLS
except ImportError:
    DATA_PROCESSED = ROOT_DIR / "data" / "03_processed" / "dataset_final.csv"
    COL_PARTICIPANT = 'participant'
    SIGNAL_COLS = ["acc_x", "acc_y", "acc_z", "eda", "wrist_hr", "ibi", "temp", "breathing_rpm"]

# 1. Chargement
df_final = pd.read_csv(DATA_PROCESSED)

# 2. Profils par participant
print("Génération des profils...")
profiles = df_final.groupby(COL_PARTICIPANT)[SIGNAL_COLS].agg(['mean', 'std']).fillna(0)
X = profiles.values
participant_ids = profiles.index

# 3. Normalisation
X_scaled = StandardScaler().fit_transform(X)

# 4. IsolationForest (Contamination 15% pour isoler ~2 sujets)
clf = IsolationForest(random_state=42, contamination=0.2)
preds = clf.fit_predict(X_scaled) 

# 5. PCA à 3 composantes
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
var_ratio = pca.explained_variance_ratio_
colors = ['red' if p == -1 else 'royalblue' for p in preds]

# ── GRAPHIQUE 1 : VUE 3D AVEC PROJECTIONS ────────────────────────────────────
print("Génération du plan 3D...")
fig_3d = plt.figure(figsize=(12, 10))
ax = fig_3d.add_subplot(111, projection='3d')
sns.set_style("white")

# Tracer les points
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colors, s=150, edgecolors='k', alpha=0.7, zorder=5)

# Lignes de projection orthogonale vers les 3 plans
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
z_min, z_max = ax.get_zlim()

for i in range(len(X_pca)):
    x, y, z = X_pca[i, 0], X_pca[i, 1], X_pca[i, 2]
    # vers le Sol (XY)
    ax.plot([x, x], [y, y], [z_min, z], linestyle='--', color='black', linewidth=0.5, alpha=0.15)
    # vers le Mur arrière (XZ)
    ax.plot([x, x], [y, y_max], [z, z], linestyle='--', color='black', linewidth=0.5, alpha=0.15)
    # vers le Mur latéral (YZ)
    ax.plot([x_min, x], [y, y], [z, z], linestyle='--', color='black', linewidth=0.5, alpha=0.15)
    
    ax.text(x, y, z, f" P{participant_ids[i]:02d}", fontsize=10, fontweight='bold')

ax.set_title(f"Représentation 3D des Participants\nVariance totale: {sum(var_ratio):.1%}", fontsize=14)
ax.set_xlabel(f"CP1 ({var_ratio[0]:.1%})")
ax.set_ylabel(f"CP2 ({var_ratio[1]:.1%})")
ax.set_zlabel(f"CP3 ({var_ratio[2]:.1%})")
ax.view_init(elev=20, azim=45)

output_3d = ROOT_DIR / "reports" / "plan_3d_participants.png"
plt.savefig(output_3d, bbox_inches='tight')
print(f"✔ Plan 3D (avec projections) sauvegardé : {output_3d}")


# ── GRAPHIQUE 2 : LES 2 PREMIERS PLANS 2D ─────────────────────────────────────
print("Génération des plans 2D...")
fig_2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
sns.set_style("whitegrid")

# Plan 1: CP1 vs CP2
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=150, edgecolors='k', alpha=0.8)
for i, txt in enumerate(participant_ids):
    ax1.annotate(f"P{txt:02d}", (X_pca[i, 0], X_pca[i, 1]), xytext=(5, 5), textcoords='offset points', fontweight='bold')
ax1.set_title(f"Plan 1 : CP1 vs CP2 ({var_ratio[0]+var_ratio[1]:.1%})", fontsize=14)
ax1.set_xlabel(f"CP1 ({var_ratio[0]:.1%})")
ax1.set_ylabel(f"CP2 ({var_ratio[1]:.1%})")

# Plan 2: CP1 vs CP3
ax2.scatter(X_pca[:, 0], X_pca[:, 2], c=colors, s=150, edgecolors='k', alpha=0.8)
for i, txt in enumerate(participant_ids):
    ax2.annotate(f"P{txt:02d}", (X_pca[i, 0], X_pca[i, 2]), xytext=(5, 5), textcoords='offset points', fontweight='bold')
ax2.set_title(f"Plan 2 : CP1 vs CP3 ({var_ratio[0]+var_ratio[2]:.1%})", fontsize=14)
ax2.set_xlabel(f"CP1 ({var_ratio[0]:.1%})")
ax2.set_ylabel(f"CP3 ({var_ratio[2]:.1%})")

output_2d = ROOT_DIR / "reports" / "plans_2d_participants.png"
plt.savefig(output_2d, bbox_inches='tight')
print(f"✔ Les 2 plans 2D sauvegardés : {output_2d}")

# 7. Résumé terminal
print("\nClassement des participants (Isolation Forest):")
for i, s_id in enumerate(participant_ids):
    status = "⚠ ATYPIQUE" if preds[i] == -1 else "Normal"
    print(f"  P{s_id:02d} : {status}")

# ── GRAPHIQUE 3 : PLAN DYNAMIQUE (HTML) ──────────────────────────────────────
try:
    import plotly.express as px
    print("\nGénération du plan dynamique interactif...")
    
    # Préparation des données
    df_pca = pd.DataFrame(X_pca, columns=['CP1', 'CP2', 'CP3'])
    df_pca['Participant'] = [f"P{i:02d}" for i in participant_ids]
    df_pca['Statut'] = ['Atypique' if p == -1 else 'Normal' for p in preds]
    
    # Création du plot Plotly
    fig_dyn = px.scatter_3d(
        df_pca, x='CP1', y='CP2', z='CP3',
        color='Statut',
        text='Participant',
        title="Analyse 3D Dynamique des Participants",
        color_discrete_map={'Normal': 'royalblue', 'Atypique': 'red'},
        opacity=0.8,
        height=700
    )
    
    # Design premium (fond sombre)
    fig_dyn.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis_title=f"CP1 ({var_ratio[0]:.1%})",
            yaxis_title=f"CP2 ({var_ratio[1]:.1%})",
            zaxis_title=f"CP3 ({var_ratio[2]:.1%})",
        )
    )
    
    output_html = ROOT_DIR / "reports" / "plan_3d_dynamique.html"
    fig_dyn.write_html(str(output_html))
    print(f"✔ Plan DYNAMIQUE sauvegardé : {output_html}")
    
except ImportError:
    print("\n[INFO] Plotly n'est pas installé. Ignore l'exportation HTML dynamique.")
