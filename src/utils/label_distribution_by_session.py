"""
Distribution des labels par session (12 participants × 3 sessions = 36 sessions)
Graphique en barres empilées : baseline / activity / fatigue
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED, COL_PARTICIPANT, COL_SESSION, COL_LABEL

# ── Chargement ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PROCESSED)

LABEL_NAMES = {-1: "baseline", 0: "activity", 1: "fatigue"}
COLORS      = {-1: "#4C72B0", 0: "#55A868", 1: "#C44E52"}  # bleu / vert / rouge

# ── Pivot : nb de samples par (participant, session, label) ──────────────────
counts = (
    df.groupby([COL_PARTICIPANT, COL_SESSION, COL_LABEL])
    .size()
    .reset_index(name="count")
)

# Toutes les combinaisons participant × session
sessions = sorted(df[[COL_PARTICIPANT, COL_SESSION]].drop_duplicates().values.tolist())
n_sessions = len(sessions)                          # 36 (ou moins si données absentes)

labels_ordered = [-1, 0, 1]
x_pos = np.arange(n_sessions)

# Matrice (3 labels × n_sessions)
data = {lbl: np.zeros(n_sessions, dtype=int) for lbl in labels_ordered}
for i, (pid, sid) in enumerate(sessions):
    sub = counts[(counts[COL_PARTICIPANT] == pid) & (counts[COL_SESSION] == sid)]
    for _, row in sub.iterrows():
        data[row[COL_LABEL]][i] = row["count"]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 7))

bottoms = np.zeros(n_sessions)
bars_handles = []
for lbl in labels_ordered:
    vals = data[lbl]
    b = ax.bar(
        x_pos, vals,
        bottom=bottoms,
        color=COLORS[lbl],
        label=LABEL_NAMES[lbl],
        edgecolor="white",
        linewidth=0.4,
        width=0.75,
    )
    bars_handles.append(b)

    # Annotation count à l'intérieur de la barre (si assez de place)
    for xi, (v, bot) in enumerate(zip(vals, bottoms)):
        if v > 80:
            ax.text(
                xi, bot + v / 2,
                str(v),
                ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold",
            )
    bottoms += vals

# ── Totaux au-dessus ─────────────────────────────────────────────────────────
for xi, total in enumerate(bottoms):
    ax.text(xi, total + 15, str(int(total)),
            ha="center", va="bottom", fontsize=7, color="#333333")

# ── Axes & labels ─────────────────────────────────────────────────────────────
xtick_labels = [f"P{pid:02d}\nS{sid}" for pid, sid in sessions]
ax.set_xticks(x_pos)
ax.set_xticklabels(xtick_labels, fontsize=8, rotation=0)

# Séparateurs verticaux entre participants
participant_ids = [s[0] for s in sessions]
for i in range(1, n_sessions):
    if participant_ids[i] != participant_ids[i - 1]:
        ax.axvline(i - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

# Labels participant au-dessus de l'axe x
unique_parts = sorted(df[COL_PARTICIPANT].unique())
part_positions = {pid: [] for pid in unique_parts}
for i, (pid, _) in enumerate(sessions):
    part_positions[pid].append(i)

for pid, idxs in part_positions.items():
    center = np.mean(idxs)
    ax.text(
        center, -350,
        f"P{pid:02d}",
        ha="center", va="top",
        fontsize=9, fontweight="bold", color="#222222",
        transform=ax.get_xaxis_transform() if False else ax.transData,
    )

ax.set_xlabel("Session  (Participant × Session)", fontsize=11, labelpad=30)
ax.set_ylabel("Nombre de samples (4 Hz)", fontsize=11)
ax.set_title(
    "Distribution des labels par session\n"
    f"({len(unique_parts)} participants × 3 sessions = {n_sessions} sessions  |  "
    f"Total : {int(bottoms.sum()):,} samples)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(title="Label", fontsize=10, title_fontsize=10, loc="upper right")
ax.set_xlim(-0.6, n_sessions - 0.4)
ax.grid(axis="y", alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)

plt.tight_layout()

# ── Sauvegarde ────────────────────────────────────────────────────────────────
out_path = Path(__file__).resolve().parent.parent.parent / "reports" / "label_distribution_by_session.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"✔ Sauvegardé : {out_path}")
