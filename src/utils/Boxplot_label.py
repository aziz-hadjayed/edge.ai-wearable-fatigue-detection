"""
Boxplot par label — analyse visuelle des features
Pour chaque signal, compare la distribution entre baseline / activity / fatigue.
Produit : reports/plots/boxplot_features.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
LABEL_NAMES = {-1: "baseline", 0: "activity", 1: "fatigue"}
PALETTE     = {"baseline": "#5B8DB8", "activity": "#F0A500", "fatigue": "#D94F3D"}
PLOTS_DIR   = BASE_DIR / "reports" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Chargement : {DATA_PROCESSED}")
    df = pd.read_csv(DATA_PROCESSED)

    # Remplacer les codes numériques par les noms de labels
    df["label_name"] = df[COL_LABEL].map(LABEL_NAMES)

    n_signals = len(SIGNAL_COLS)
    n_cols    = 2
    n_rows    = (n_signals + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 4),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, col in enumerate(SIGNAL_COLS):
        ax = axes[i]

        sns.boxplot(
            data      = df,
            x         = "label_name",
            y         = col,
            hue       = "label_name",
            order     = ["baseline", "activity", "fatigue"],
            palette   = PALETTE,
            width     = 0.5,
            linewidth = 1.2,
            flierprops= dict(marker="o", markersize=2, alpha=0.3),
            legend    = False,
            ax        = ax,
        )

        # Médiane annotée sur chaque boîte
        for j, lbl in enumerate(["baseline", "activity", "fatigue"]):
            median = df.loc[df["label_name"] == lbl, col].median()
            ax.text(j, median, f" {median:.2f}",
                    va="center", ha="left", fontsize=7.5, color="black")

        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("valeur normalisée", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Masquer les axes vides si nombre impair
    for j in range(n_signals, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution des features par label\n(baseline vs activity vs fatigue)",
                 fontsize=14, fontweight="bold")

    out = PLOTS_DIR / "boxplot_features.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Sauvegardé : {out}")

    # ── Résumé console : médiane par label ────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Feature':<18} {'baseline':>12} {'activity':>12} {'fatigue':>12}")
    print("="*65)
    for col in SIGNAL_COLS:
        vals = [df.loc[df["label_name"] == lbl, col].median()
                for lbl in ["baseline", "activity", "fatigue"]]
        print(f"{col:<18} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f}")


if __name__ == "__main__":
    main()
