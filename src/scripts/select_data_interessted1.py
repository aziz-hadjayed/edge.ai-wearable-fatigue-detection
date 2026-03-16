import os

import pandas as pd
from tqdm import tqdm

# Définir le chemin du répertoire contenant les fichiers
data_dir = "/home/aziz/Desktop/fatigue_detection/datasets/archive/fatigueset"
output_dir = "/home/aziz/Desktop/fatigue_detection/datasets"

# Définir les fichiers que vous voulez combiner
files_to_combine = [
    "chest_physiology_summary.csv",
    "exp_fatigue.csv",
    "exp_markers.csv",
    "wrist_acc.csv",
    "wrist_eda.csv",
    "wrist_hr.csv",
    "wrist_ibi.csv",
    "wrist_skin_temperature.csv",
]

# Créer une liste pour stocker les DataFrames combinés
combined_data = []

# Itérer sur chaque participant
for participant in tqdm(range(1, 13)):
    participant_data = None

    # Itérer sur chaque session pour le participant
    for session in range(1, 4):
        session_data = None

        # Itérer sur chaque fichier à combiner
        for file_name in files_to_combine:
            # Construire le chemin complet du fichier
            file_path = os.path.join(
                data_dir, f"{participant:02d}/{session:02d}/{file_name}"
            )

            # Lire le fichier CSV
            df = pd.read_csv(file_path)

            # Ajouter une colonne indiquant le participant et la session
            df["participant"] = participant
            df["session"] = session

            # Si c'est le premier fichier pour ce participant et cette session, stocker les données
            if session_data is None:
                session_data = df
            else:
                # Sinon, combiner les données
                session_data = pd.concat([session_data, df], ignore_index=True)

        # Si c'est le premier fichier pour ce participant, stocker les données
        if participant_data is None:
            participant_data = session_data
        else:
            # Sinon, combiner les données
            participant_data = pd.concat(
                [participant_data, session_data], ignore_index=True
            )

    # Ajouter les données du participant à la liste des données combinées
    combined_data.append(participant_data)


# Concaténer toutes les données des participants
data_selected = pd.concat(combined_data, ignore_index=True)

# Sauvegarder les données combinées dans un nouveau fichier CSV
output_path = os.path.join(output_dir, "data_selected.csv")
data_selected.to_csv(output_path, index=False)

print(f"Données combinées sauvegardées à {output_path}")
