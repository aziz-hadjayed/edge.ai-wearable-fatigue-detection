"""
Distribution des labels par session (participant x session).
Graphique en barres empilees pour tous les labels presents dans le dataset.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from config import (  # noqa: E402
    COL_LABEL,
    COL_PARTICIPANT,
    COL_SESSION,
    OUTPUT_PATH_NO_SMOTE,
    OUTPUT_PATH_SMOTE,
)

# Charger le dataset SMOTE si disponible, sinon le dataset sans SMOTE.
DATASET_PATH = OUTPUT_PATH_SMOTE if OUTPUT_PATH_SMOTE.exists() else OUTPUT_PATH_NO_SMOTE
OUTPUT_PATH = root_dir / "reports" / "label_distribution_by_session.png"

LABEL_NAMES = {
    -1: "baseline",
    0: "activity",
    1: "pre_fatigue",
    2: "fatigue_light",
    3: "fatigue",
}
COLORS = {
    -1: "#4C72B0",
    0: "#55A868",
    1: "#F2C14E",
    2: "#F58518",
    3: "#C44E52",
}


def _sort_key(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _format_id(prefix, value):
    try:
        return f"{prefix}{int(value):02d}"
    except (TypeError, ValueError):
        return f"{prefix}{value}"


def main():
    df = pd.read_csv(DATASET_PATH)

    labels_ordered = sorted(df[COL_LABEL].dropna().unique(), key=_sort_key)
    sessions = (
        df[[COL_PARTICIPANT, COL_SESSION]]
        .drop_duplicates()
        .sort_values([COL_PARTICIPANT, COL_SESSION], key=lambda col: col.map(_sort_key))
        .itertuples(index=False, name=None)
    )
    sessions = list(sessions)
    n_sessions = len(sessions)

    counts = (
        df.groupby([COL_PARTICIPANT, COL_SESSION, COL_LABEL])
        .size()
        .reset_index(name="count")
    )

    x_pos = np.arange(n_sessions)
    data = {label: np.zeros(n_sessions, dtype=int) for label in labels_ordered}

    for i, (participant_id, session_id) in enumerate(sessions):
        mask = (
            (counts[COL_PARTICIPANT] == participant_id)
            & (counts[COL_SESSION] == session_id)
        )
        for _, row in counts.loc[mask].iterrows():
            data[row[COL_LABEL]][i] = int(row["count"])

    fig, ax = plt.subplots(figsize=(22, 8))
    bottoms = np.zeros(n_sessions, dtype=int)

    for label in labels_ordered:
        values = data[label]
        color = COLORS.get(label, "#888888")
        label_name = LABEL_NAMES.get(label, f"label {label}")
        bars = ax.bar(
            x_pos,
            values,
            bottom=bottoms,
            color=color,
            label=label_name,
            edgecolor="white",
            linewidth=0.4,
            width=0.75,
        )

        for xi, (value, bottom) in enumerate(zip(values, bottoms)):
            if value > 80:
                ax.text(
                    xi,
                    bottom + value / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    fontweight="bold",
                )
        bottoms += values

    for xi, total in enumerate(bottoms):
        ax.text(
            xi,
            total + max(bottoms.max() * 0.01, 15),
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )

    xtick_labels = [
        f"{_format_id('P', participant_id)}{_sort_key(session_id)}"
        for participant_id, session_id in sessions
    ]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xtick_labels, fontsize=8, rotation=0)

    participant_ids = [participant_id for participant_id, _ in sessions]
    for i in range(1, n_sessions):
        if participant_ids[i] != participant_ids[i - 1]:
            ax.axvline(i - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlabel("Session (Participant x Session)", fontsize=11)
    ax.set_ylabel("Nombre de samples", fontsize=11)
    ax.set_title(
        "Distribution des labels par session"
        f"{n_sessions} sessions | Total : {int(bottoms.sum()):,} samples | {DATASET_PATH.name}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Label", fontsize=10, title_fontsize=10, loc="upper right")
    ax.set_xlim(-0.6, n_sessions - 0.4)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Sauvegarde : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
