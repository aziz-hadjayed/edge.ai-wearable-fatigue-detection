import pandas as pd
import matplotlib.pyplot as plt
import os

# Lire les données du fichier csv
ecg_data = pd.read_csv('/home/aziz/Desktop/fatigue_detection/src/scripts/01/chest_raw_ecg.csv')
output_path = "/home/aziz/Desktop/fatigue_detection/assets/ecg_poitrine_01-01/p01_s01_ecg.png"

# Extraire les colonnes timestamp et ecg_waveform
time = ecg_data['timestamp']
ecg_waveform = ecg_data['ecg_waveform']

# Convertir les timestamps en secondes
time = (time - time[0]) / 1000

# Tracer le courbe des valeurs ECG en fonction du temps
plt.plot(time, ecg_waveform, label='ECG Waveform')

# Ajouter les étiquettes et la légende
plt.xlabel('Time (s)')
plt.ylabel('ECG Waveform')
plt.title('ECG Waveform Over Time')
plt.legend()

# Sauvegarde
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Graphique sauvegardé sous : {output_path}")
