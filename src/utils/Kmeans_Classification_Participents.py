import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Chemins ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from src.config import DATA_PROCESSED, SIGNAL_COLS, COL_PARTICIPANT

# ── 1. Chargement ──────────────────────────────────────────────────────────────
print("Chargement des données...")
df = pd.read_csv(DATA_PROCESSED)
print(f"  Shape : {df.shape} | Participants : {df[COL_PARTICIPANT].nunique()}")

# ── 2. Profil par participant : mean + std de chaque feature ───────────────────
profiles = df.groupby(COL_PARTICIPANT)[SIGNAL_COLS].agg(["mean", "std"]).fillna(0)
profiles.columns = [f"{col}_{stat}" for col, stat in profiles.columns]
participant_ids = profiles.index.tolist()
X_raw = profiles.values

# ── 3. Normalisation ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ── 4. KMeans K = 2 ────────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_scaled)

# Nommage des groupes (0 → "Groupe A", 1 → "Groupe B")
group_names = {0: "Groupe A", 1: "Groupe B"}
group_colors = {"Groupe A": "royalblue", "Groupe B": "tomato"}

# ── 5. Tableau complet des features par participant ────────────────────────────
df_result = profiles.copy()
df_result.insert(0, "Cluster", [group_names[l] for l in labels])
df_result.index = [f"P{p:02d}" for p in participant_ids]

print("\n" + "=" * 100)
print("CLASSIFICATION K-MEANS (K=2) — TOUTES LES FEATURES PAR PARTICIPANT")
print("=" * 100)

# Affichage par groupe
for grp in ["Groupe A", "Groupe B"]:
    subset = df_result[df_result["Cluster"] == grp].drop(columns="Cluster")
    participants_in_group = df_result[df_result["Cluster"] == grp].index.tolist()
    print(f"\n{'─'*100}")
    print(f"  {grp}  ({len(participants_in_group)} participants) : {', '.join(participants_in_group)}")
    print(f"{'─'*100}")
    # Afficher mean et std séparément pour plus de lisibilité
    mean_cols = [c for c in subset.columns if c.endswith("_mean")]
    std_cols  = [c for c in subset.columns if c.endswith("_std")]

    print("\n  [MOYENNES]")
    mean_df = subset[mean_cols].copy()
    mean_df.columns = [c.replace("_mean", "") for c in mean_cols]
    print(mean_df.round(4).to_string())

    print("\n  [ÉCARTS-TYPES]")
    std_df = subset[std_cols].copy()
    std_df.columns = [c.replace("_std", "") for c in std_cols]
    print(std_df.round(4).to_string())

# Résumé terminal
print("\n" + "=" * 100)
print("RÉSUMÉ DE LA CLASSIFICATION")
print("=" * 100)
for grp in ["Groupe A", "Groupe B"]:
    members = df_result[df_result["Cluster"] == grp].index.tolist()
    print(f"  {grp}  ({len(members)}) : {', '.join(members)}")

# ── 6. Heatmap des features (moyennes par participant) ─────────────────────────
print("\nGénération des graphiques...")

mean_cols = [c for c in df_result.columns if c.endswith("_mean")]
heat_data = df_result[mean_cols].copy()
heat_data.columns = [c.replace("_mean", "") for c in mean_cols]
heat_data_scaled = pd.DataFrame(
    StandardScaler().fit_transform(heat_data),
    index=heat_data.index,
    columns=heat_data.columns,
)
cluster_order = df_result.sort_values("Cluster").index
heat_data_scaled = heat_data_scaled.loc[cluster_order]
row_colors = [group_colors[df_result.loc[p, "Cluster"]] for p in cluster_order]

fig_heat, ax_heat = plt.subplots(figsize=(14, 7))
sns.heatmap(
    heat_data_scaled,
    ax=ax_heat,
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Z-score"},
)
ax_heat.set_title("Heatmap des Features (moyennes normalisées) — KMeans K=2", fontsize=14, fontweight="bold")
ax_heat.set_xlabel("Features")
ax_heat.set_ylabel("Participants")

# Colorier les labels Y selon le groupe
for tick_label in ax_heat.get_yticklabels():
    p = tick_label.get_text()
    tick_label.set_color(group_colors[df_result.loc[p, "Cluster"]])
    tick_label.set_fontweight("bold")

plt.tight_layout()
out_heat = ROOT_DIR / "reports" / "kmeans_heatmap_participants.png"
fig_heat.savefig(out_heat, dpi=150, bbox_inches="tight")
print(f"  ✔ Heatmap sauvegardée : {out_heat}")

# ── 7. PCA 2D — visualisation des clusters ─────────────────────────────────────
pca = PCA(n_components=2)
X_pca2 = pca.fit_transform(X_scaled)
var2 = pca.explained_variance_ratio_

fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
for grp in ["Groupe A", "Groupe B"]:
    idx = [i for i, l in enumerate(labels) if group_names[l] == grp]
    ax_2d.scatter(
        X_pca2[idx, 0], X_pca2[idx, 1],
        label=grp, color=group_colors[grp],
        s=200, edgecolors="k", alpha=0.85, zorder=5,
    )
for i, pid in enumerate(participant_ids):
    ax_2d.annotate(
        f"P{pid:02d}",
        (X_pca2[i, 0], X_pca2[i, 1]),
        xytext=(6, 4), textcoords="offset points",
        fontsize=10, fontweight="bold",
    )

# Centroïdes projetés
centroids_pca = pca.transform(kmeans.cluster_centers_)
for k, grp in group_names.items():
    ax_2d.scatter(
        centroids_pca[k, 0], centroids_pca[k, 1],
        marker="X", s=400, color=group_colors[grp],
        edgecolors="k", linewidths=1.5, zorder=10, label=f"Centroïde {grp}",
    )

ax_2d.set_title(
    f"KMeans K=2 — Projection PCA 2D\nVariance expliquée : {sum(var2):.1%}",
    fontsize=13, fontweight="bold",
)
ax_2d.set_xlabel(f"CP1 ({var2[0]:.1%})")
ax_2d.set_ylabel(f"CP2 ({var2[1]:.1%})")
ax_2d.legend(loc="best")
ax_2d.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

out_2d = ROOT_DIR / "reports" / "kmeans_pca2d_participants.png"
fig_2d.savefig(out_2d, dpi=150, bbox_inches="tight")
print(f"  ✔ PCA 2D sauvegardée   : {out_2d}")

# ── 8. PCA 3D — visualisation des clusters ─────────────────────────────────────
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)
var3 = pca3.explained_variance_ratio_

fig_3d = plt.figure(figsize=(12, 10))
ax_3d = fig_3d.add_subplot(111, projection="3d")

for grp in ["Groupe A", "Groupe B"]:
    idx = [i for i, l in enumerate(labels) if group_names[l] == grp]
    ax_3d.scatter(
        X_pca3[idx, 0], X_pca3[idx, 1], X_pca3[idx, 2],
        label=grp, color=group_colors[grp],
        s=150, edgecolors="k", alpha=0.85, zorder=5,
    )

for i, pid in enumerate(participant_ids):
    ax_3d.text(X_pca3[i, 0], X_pca3[i, 1], X_pca3[i, 2], f" P{pid:02d}", fontsize=9, fontweight="bold")

ax_3d.set_title(
    f"KMeans K=2 — Projection PCA 3D\nVariance totale : {sum(var3):.1%}",
    fontsize=13, fontweight="bold",
)
ax_3d.set_xlabel(f"CP1 ({var3[0]:.1%})")
ax_3d.set_ylabel(f"CP2 ({var3[1]:.1%})")
ax_3d.set_zlabel(f"CP3 ({var3[2]:.1%})")
ax_3d.legend(loc="best")
ax_3d.view_init(elev=20, azim=45)
plt.tight_layout()

out_3d = ROOT_DIR / "reports" / "kmeans_pca3d_participants.png"
fig_3d.savefig(out_3d, dpi=150, bbox_inches="tight")
print(f"  ✔ PCA 3D sauvegardée   : {out_3d}")

# ── 9. Radar chart — profil moyen de chaque groupe ────────────────────────────
features_radar = [c.replace("_mean", "") for c in mean_cols]
N = len(features_radar)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # fermer le polygone

fig_radar, ax_radar = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for grp in ["Groupe A", "Groupe B"]:
    idx = [i for i, l in enumerate(labels) if group_names[l] == grp]
    vals = heat_data_scaled.iloc[
        [list(cluster_order).index(f"P{participant_ids[i]:02d}") for i in idx]
    ].mean().values.tolist()
    vals += vals[:1]
    ax_radar.plot(angles, vals, color=group_colors[grp], linewidth=2, label=grp)
    ax_radar.fill(angles, vals, color=group_colors[grp], alpha=0.15)

ax_radar.set_thetagrids(np.degrees(angles[:-1]), features_radar, fontsize=9)
ax_radar.set_title("Profil moyen des groupes KMeans (Z-score)", fontsize=13, fontweight="bold", pad=20)
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()

out_radar = ROOT_DIR / "reports" / "kmeans_radar_groupes.png"
fig_radar.savefig(out_radar, dpi=150, bbox_inches="tight")
print(f"  ✔ Radar sauvegardé     : {out_radar}")

# ── 10. Plan dynamique HTML (Plotly) ──────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go

    df_pca3 = pd.DataFrame(X_pca3, columns=["CP1", "CP2", "CP3"])
    df_pca3["Participant"] = [f"P{p:02d}" for p in participant_ids]
    df_pca3["Groupe"] = [group_names[l] for l in labels]

    # Ajouter toutes les features comme hover
    for col in profiles.columns:
        df_pca3[col] = profiles[col].values

    hover_cols = list(profiles.columns)

    fig_dyn = px.scatter_3d(
        df_pca3,
        x="CP1", y="CP2", z="CP3",
        color="Groupe",
        text="Participant",
        title="KMeans K=2 — Classification des 12 Participants (3D Interactif)",
        color_discrete_map={"Groupe A": "royalblue", "Groupe B": "tomato"},
        hover_data={"Participant": True, "Groupe": True, "CP1": ":.3f", "CP2": ":.3f", "CP3": ":.3f",
                    **{c: ":.4f" for c in hover_cols}},
        opacity=0.85,
        height=750,
    )
    fig_dyn.update_traces(textposition="top center", marker=dict(size=8, line=dict(width=1, color="black")))
    fig_dyn.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=60),
        scene=dict(
            xaxis_title=f"CP1 ({var3[0]:.1%})",
            yaxis_title=f"CP2 ({var3[1]:.1%})",
            zaxis_title=f"CP3 ({var3[2]:.1%})",
        ),
        legend_title="Groupe KMeans",
    )

    out_html = ROOT_DIR / "reports" / "kmeans_3d_dynamique.html"
    fig_dyn.write_html(str(out_html))
    print(f"  ✔ Plan dynamique HTML  : {out_html}")

except ImportError:
    print("  [INFO] Plotly non installé — plan dynamique ignoré.")

# ── 11. Export CSV ─────────────────────────────────────────────────────────────
out_csv = ROOT_DIR / "reports" / "kmeans_participants_features.csv"
df_result.to_csv(out_csv)
print(f"  ✔ CSV exporté          : {out_csv}")

plt.show()
print("\n✔ TERMINÉ")
