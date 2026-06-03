import os
import pandas as pd
from tabulate import tabulate

# Chemin du dossier contenant les fichiers CSV
DATA_DIR = "/media/mohamedaziz-hadjayed/D/aziz_data/fatigue_detection/edge-ai-wearable-fatigue-detection/data/01_raw/01/02"

def get_sampling_frequency(file_path):
    """Calcule la fréquence d'échantillonnage d'un fichier CSV"""
    try:
        df = pd.read_csv(file_path)

        # Vérifier si la colonne 'timestamp' existe
        if 'timestamp' in df.columns:
            # Convertir les timestamps en datetime si ce sont des entiers
            if pd.api.types.is_integer_dtype(df['timestamp']):
                # Supposons que les timestamps sont en millisecondes
                timestamps = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                timestamps = pd.to_datetime(df['timestamp'])

            time_diff = timestamps.diff().dropna().mean()
            return 1 / time_diff.total_seconds() if time_diff.total_seconds() > 0 else None
        else:
            # Si pas de colonne timestamp, on suppose que les données sont uniformément espacées
            return len(df) / (df.index[-1] - df.index[0]).total_seconds() if len(df) > 1 else None
    except Exception as e:
        print(f"Erreur lors de la lecture de {file_path}: {e}")
        return None


def main():
    # Liste des fichiers CSV dans le dossier
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    # Calcul des fréquences d'échantillonnage
    results = []
    for file in csv_files:
        file_path = os.path.join(DATA_DIR, file)
        freq = get_sampling_frequency(file_path)
        results.append({
            'Fichier': file,
            'Fréquence (Hz)': f"{freq:.2f}" if freq is not None else "Inconnue"
        })

    # Affichage des résultats dans un tableau
    print(tabulate(results, headers='keys', tablefmt='grid'))

if __name__ == "__main__":
    main()
