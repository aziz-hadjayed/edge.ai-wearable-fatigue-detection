import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# Lire le fichier CSV (Utilisation du capteur d'oreille PPG qui contient la colonne 'green')
file_path = '/home/aziz/Desktop/fatigue_detection/datasets/archive/fatigueset/01/01/ear_ppg_left.csv'
if not os.path.exists(file_path):
    print(f"Erreur : Le fichier {file_path} est introuvable.")
    exit()

df = pd.read_csv(file_path)

# Convertir la colonne 'timestamp' en datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Calculer le temps de début
start_time = df['timestamp'].min()

# Convertir la colonne 'timestamp' en secondes relatives
df['seconds'] = (df['timestamp'] - start_time).dt.total_seconds()

# Sélectionner la colonne PPG (la colonne 'green' est souvent utilisée pour l'Estimation HR)
ppg_data = df['green'].values
# Soustraire la moyenne pour supprimer la composante DC
ppg_data = ppg_data - np.mean(ppg_data)

# Calculer la fréquence d'échantillonnage (fs)
# On calcule la différence moyenne entre les timestamps
time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
fs = 1.0 / time_diffs.mean()
print(f"Fréquence d'échantillonnage estimée : {fs:.2f} Hz")

# Calculer la transformée de Fourier
n = len(ppg_data)
fft_result = fft(ppg_data)
fft_freq = fftfreq(n, d=1/fs)

# On ne garde que les fréquences positives
positive_freq_mask = fft_freq > 0
fft_freq = fft_freq[positive_freq_mask]
fft_abs = np.abs(fft_result[positive_freq_mask])

# Trouver la fréquence cardiaque
# On considère que la fréquence cardiaque est généralement entre 40 et 180 battements par minute (BPM)
hr_range = (40, 180)
hr_range_hz = (hr_range[0] / 60, hr_range[1] / 60)

# Trouver l'index des fréquences dans la plage HR
hr_indices = np.where((fft_freq >= hr_range_hz[0]) & (fft_freq <= hr_range_hz[1]))

# Trouver l'indice avec l'amplitude maximale dans la plage HR
if len(hr_indices[0]) > 0:
    max_index_in_range = np.argmax(fft_abs[hr_indices])
    max_index = hr_indices[0][max_index_in_range]
    
    # Calculer la fréquence cardiaque en BPM
    hr = fft_freq[max_index] * 60
    print(f"Fréquence cardiaque estimée : {hr:.2f} BPM")
else:
    print("Impossible de trouver une fréquence cardiaque dans la plage spécifiée.")
    hr = 0

# Tracer la transformée de Fourier
plt.figure(figsize=(10, 6))
plt.plot(fft_freq, fft_abs)
plt.axvline(x=hr/60, color='r', linestyle='--', label=f'HR Estimé: {hr:.2f} BPM')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')
plt.title('Analyse Spectrale du signal PPG (FFT)')
plt.xlim(0, 5) # Zoom sur les fréquences basses (0-5 Hz covers up to 300 BPM)
plt.legend()
plt.grid(True)

# Sauvegarde du graphique
output_path = '/home/aziz/Desktop/fatigue_detection/assets/ppg/fft_ppg_P01_S01.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Graphique sauvegardé sous : {output_path}")






# Tracer la fréquence cardiaque en fonction du temps
plt.figure(figsize=(10, 6))
plt.plot(df['seconds'], df['green'], label='Signal PPG')
plt.scatter(df['seconds'], df['green'], label=f'HR estimée : {hr:.2f} BPM')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal PPG et Fréquence cardiaque estimée')
plt.legend()
plt.grid(True)
# Sauvegarde du graphique
output_path = '/home/aziz/Desktop/fatigue_detection/assets/ppg/ppg_P01_S01.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Graphique sauvegardé sous : {output_path}")

# plt.show() # Commenté pour éviter de bloquer l'exécution en mode script
