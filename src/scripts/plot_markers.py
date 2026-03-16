import os

import matplotlib.pyplot as plt
import pandas as pd

# Chemins des fichiers
csv_path = "/home/aziz/Desktop/fatigue_detection/datasets/archive/fatigueset/01/01/exp_markers.csv"
output_path = "/home/aziz/Desktop/fatigue_detection/assets/markers/p01_s01_markers.png"

# Lecture des données
df = pd.read_csv(csv_path)

# Nettoyage : suppression des lignes sans timestamp
df = df.dropna(subset=["utcTime"])

# Conversion des timestamps (ms) en datetime
df["datetime"] = pd.to_datetime(df["utcTime"], unit="ms")

# Calcul du temps écoulé depuis le début (en minutes)
start_time = df["datetime"].min()
df["minutes"] = (df["datetime"] - start_time).dt.total_seconds() / 60

# Création de la figure
plt.figure(figsize=(12, 6))
plt.hlines(
    y=range(len(df)),
    xmin=0,
    xmax=df["minutes"].max(),
    colors="gray",
    linestyles="--",
    alpha=0.3,
)

# Définition des couleurs
colors = {
    "start_baseline": "green",
    "end_baseline": "green",
    "start_activity": "blue",
    "end_activity": "blue",
    "start_fatigue": "red",
    "end_fatigue": "red",
}

# Coloriage des points
for i, txt in enumerate(df["eventMarker"]):
    if txt in colors:
        plt.scatter(df["minutes"].iloc[i], i, color=colors[txt], s=100, zorder=3)

# Ajout des labels
for i, txt in enumerate(df["eventMarker"]):
    plt.annotate(
        txt,
        (df["minutes"].iloc[i], i),
        xytext=(10, -5),
        textcoords="offset points",
        fontsize=9,
    )

plt.yticks(range(len(df)), df["eventMarker"], fontsize=8)
plt.xlabel("Temps écoulé (minutes)")
plt.title("Chronologie des Événements - Participant 01, Session 01")
plt.grid(axis="x", linestyle="-", alpha=0.2)
plt.tight_layout()

# Sauvegarde
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Graphique sauvegardé sous : {output_path}")


u
