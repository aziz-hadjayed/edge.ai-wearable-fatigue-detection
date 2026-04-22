"""
analyze_dataset.py
==================
Analyse complète du fichier dataset_balanced.csv.

Ce script génère 6 figures sauvegardées dans reports/analysis/ :
  1. label_distribution.png      — distribution globale + par participant
  2. signal_boxplots.png         — boxplots des signaux par label (-1 / 0 / 1)
  3. correlation_heatmap.png     — matrice de corrélation des signaux
  4. signal_per_participant.png  — moyenne de chaque signal par participant
  5. support_per_fold.png        — support (nb samples) par label × participant
  6. timeseries_sample.png       — exemple de série temporelle (P01, S01)

Usage :
    python3 src/analyze_dataset.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ── Chemins ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "03_processed" / "dataset_balanced.csv"
OUT_DIR    = BASE_DIR / "reports" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constantes ───────────────────────────────────────────────────────────────
SIGNAL_COLS = ["acc_x", "acc_y", "acc_z", "eda", "wrist_hr",
               "ibi", "temp", "breathing_rpm", "age", "gender"]
LABEL_NAMES = {-1: "baseline", 0: "activity", 1: "fatigue"}
LABEL_COLORS = {-1: "#4A90D9", 0: "#27AE60", 1: "#E74C3C"}
PALETTE = [LABEL_COLORS[k] for k in [-1, 0, 1]]

# ── Chargement ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("ANALYSE DU DATASET")
print(f"{'='*60}")
print(f"Chargement de : {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df["label_name"] = df["label"].map(LABEL_NAMES)

# ── 0. Résumé textuel ────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print("0. APERÇU GÉNÉRAL")
print(f"{'─'*50}")
print(f"  Shape              : {df.shape}")
print(f"  Participants       : {sorted(df['participant'].unique())}")
print(f"  Sessions/participant: {df.groupby('participant')['session'].nunique().to_dict()}")
print(f"  NaN                : {df.isnull().sum().sum()}")
print(f"\n  Distribution des labels :")
dist = df["label"].value_counts().sort_index()
total = len(df)
for lbl, cnt in dist.items():
    print(f"    {LABEL_NAMES[lbl]:>10}  ({lbl:+d})  : {cnt:6d}  ({cnt/total*100:.1f}%)")

print(f"\n  Statistiques par signal :")
print(df[SIGNAL_COLS].describe().round(3).to_string())

print(f"\n  Samples par participant :")
print(df.groupby("participant")["label"].count().to_string())

print(f"\n  Support (label × participant) :")
pivot = df.groupby(["participant", "label"])["label"].count().unstack().fillna(0).astype(int)
pivot.columns = [LABEL_NAMES[c] for c in pivot.columns]
print(pivot.to_string())


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Distribution des labels
# ═══════════════════════════════════════════════════════════════════════════
print("\n\n→ Figure 1 : Distribution des labels …")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Distribution des labels", fontsize=15, fontweight="bold")

# 1a — distribution globale (barplot)
ax = axes[0]
counts = df["label"].value_counts().sort_index()
bars = ax.bar(
    [LABEL_NAMES[k] for k in counts.index],
    counts.values,
    color=PALETTE, edgecolor="white", linewidth=1.2
)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:,}\n({val/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=10)
ax.set_title("Distribution globale")
ax.set_ylabel("Nombre de samples")
ax.set_ylim(0, max(counts.values) * 1.18)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

# 1b — distribution par participant (stacked bar)
ax = axes[1]
pivot_pct = df.groupby(["participant", "label_name"]).size().unstack(fill_value=0)
pivot_pct = pivot_pct[["baseline", "activity", "fatigue"]]
colors_map = {"baseline": LABEL_COLORS[-1], "activity": LABEL_COLORS[0], "fatigue": LABEL_COLORS[1]}
bottom = np.zeros(len(pivot_pct))
for lbl in ["baseline", "activity", "fatigue"]:
    vals = pivot_pct[lbl].values
    ax.bar(pivot_pct.index.astype(str), vals, bottom=bottom,
           label=lbl, color=colors_map[lbl], edgecolor="white", linewidth=0.8)
    bottom += vals
ax.set_title("Distribution par participant (stacked)")
ax.set_xlabel("Participant")
ax.set_ylabel("Nombre de samples")
ax.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
p = OUT_DIR / "label_distribution.png"
plt.savefig(p, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  ✔ Sauvegardé : {p}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Boxplots des signaux par label
# ═══════════════════════════════════════════════════════════════════════════
print("→ Figure 2 : Boxplots par label …")
n_signals = len(SIGNAL_COLS)
ncols = 5
nrows = (n_signals + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
fig.suptitle("Distribution des signaux par label", fontsize=15, fontweight="bold")
axes_flat = axes.flatten()

for idx, col in enumerate(SIGNAL_COLS):
    ax = axes_flat[idx]
    groups = [df[df["label"] == lbl][col].dropna().values for lbl in [-1, 0, 1]]
    bp = ax.boxplot(groups, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2},
                    whiskerprops={"linewidth": 1.2},
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title(col, fontsize=11, fontweight="bold")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["baseline", "activity", "fatigue"], fontsize=8, rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

for idx in range(n_signals, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
p = OUT_DIR / "signal_boxplots.png"
plt.savefig(p, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  ✔ Sauvegardé : {p}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Matrice de corrélation
# ═══════════════════════════════════════════════════════════════════════════
print("→ Figure 3 : Matrice de corrélation …")
fig, ax = plt.subplots(figsize=(11, 9))
corr = df[SIGNAL_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, ax=ax, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor="white",
    annot_kws={"size": 8}
)
ax.set_title("Matrice de corrélation — signaux", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
p = OUT_DIR / "correlation_heatmap.png"
plt.savefig(p, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  ✔ Sauvegardé : {p}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Moyenne des signaux par participant
# ═══════════════════════════════════════════════════════════════════════════
print("→ Figure 4 : Signaux moyens par participant …")
numeric_signals = [c for c in SIGNAL_COLS if c not in ("age", "gender")]
part_means = df.groupby("participant")[numeric_signals].mean()

nrows = len(numeric_signals)
fig, axes = plt.subplots(nrows, 1, figsize=(14, nrows * 2.2), sharex=True)
fig.suptitle("Moyenne des signaux par participant", fontsize=14, fontweight="bold")

x = np.arange(len(part_means))
for ax, col in zip(axes, numeric_signals):
    ax.bar(x, part_means[col].values, color="#5B8DE4", edgecolor="white", linewidth=0.8)
    ax.set_ylabel(col, fontsize=9, rotation=0, labelpad=60, va="center")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels([f"P{p:02d}" for p in part_means.index], fontsize=9)
axes[-1].set_xlabel("Participant")
plt.tight_layout()
p = OUT_DIR / "signal_per_participant.png"
plt.savefig(p, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  ✔ Sauvegardé : {p}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Support (samples) par label × participant
# ═══════════════════════════════════════════════════════════════════════════
print("→ Figure 5 : Support par label × participant …")
support = df.groupby(["participant", "label"]).size().unstack(fill_value=0)
support.columns = [LABEL_NAMES[c] for c in support.columns]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Support par label × participant", fontsize=14, fontweight="bold")

# 5a — heatmap
ax = axes[0]
sns.heatmap(support, ax=ax, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, linecolor="white", cbar_kws={"label": "nb samples"})
ax.set_title("Heatmap du support")
ax.set_xlabel("Label")
ax.set_ylabel("Participant")

# 5b — barplot groupé
ax = axes[1]
x = np.arange(len(support))
w = 0.25
for i, (lbl, color) in enumerate(zip(["baseline", "activity", "fatigue"], PALETTE)):
    ax.bar(x + i * w, support[lbl].values, width=w,
           label=lbl, color=color, edgecolor="white", linewidth=0.8)
ax.set_xticks(x + w)
ax.set_xticklabels([f"P{p:02d}" for p in support.index], rotation=45)
ax.set_title("Barplot groupé du support")
ax.set_ylabel("Nombre de samples")
ax.legend(title="Label")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
p = OUT_DIR / "support_per_fold.png"
plt.savefig(p, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  ✔ Sauvegardé : {p}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Exemple de série temporelle (P01, S01)
# ═══════════════════════════════════════════════════════════════════════════
print("→ Figure 6 : Exemple de série temporelle (P01, S01) …")
df_ex = df[(df["participant"] == 1) & (df["session"] == 1)].copy()
df_ex = df_ex.sort_values("timestamp").reset_index(drop=True)

plot_sigs = ["acc_x", "acc_y", "acc_z", "eda", "wrist_hr", "ibi", "temp", "breathing_rpm"]
n = len(plot_sigs)
fig, axes = plt.subplots(n + 1, 1, figsize=(16, n * 2 + 2.5), sharex=True)
fig.suptitle("Série temporelle — Participant 01, Session 01", fontsize=14, fontweight="bold")

# Zones colorées selon le label
label_colors_bg = {-1: "#D6EAF8", 0: "#D5F5E3", 1: "#FADBD8"}
t = df_ex["timestamp"].values

for ax, sig in zip(axes[:-1], plot_sigs):
    ax.plot(t, df_ex[sig].values, linewidth=0.8, color="#333333")
    # Colorer le fond par label
    prev_lbl = df_ex["label"].iloc[0]
    start_t  = t[0]
    for i in range(1, len(df_ex)):
        curr_lbl = df_ex["label"].iloc[i]
        if curr_lbl != prev_lbl or i == len(df_ex) - 1:
            ax.axvspan(start_t, t[i], alpha=0.25, color=label_colors_bg[prev_lbl], lw=0)
            start_t  = t[i]
            prev_lbl = curr_lbl
    ax.set_ylabel(sig, fontsize=8, rotation=0, labelpad=60, va="center")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

# Dernière ligne : le label lui-même
ax_lbl = axes[-1]
lbl_numeric = df_ex["label"].values
ax_lbl.fill_between(t, lbl_numeric, step="mid",
                     color="#7D3C98", alpha=0.6)
ax_lbl.plot(t, lbl_numeric, linewidth=0.6, color="#7D3C98")
ax_lbl.set_yticks([-1, 0, 1])
ax_lbl.set_yticklabels(["baseline", "activity", "fatigue"], fontsize=8)
ax_lbl.set_ylabel("label", fontsize=8, rotation=0, labelpad=60, va="center")
ax_lbl.set_xlabel("Timestamp (ms)")
ax_lbl.grid(alpha=0.2)
ax_lbl.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
p = OUT_DIR / "timeseries_sample.png"
plt.savefig(p, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  ✔ Sauvegardé : {p}")


# ── Résumé final ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"ANALYSE TERMINÉE — Figures sauvegardées dans : {OUT_DIR}")
print(f"{'='*60}\n")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  📊 {f.name}")
