import pandas as pd
import matplotlib.pyplot as plt
import os

# Paramètres des participants et des sessions
num_participants = 12
num_sessions = 3

for p in range(1, num_participants + 1):
    participant_id = f"{p:02d}"
    for s in range(1, num_sessions + 1):
        session_id = f"{s:02d}"
        
        # Chemin du fichier CSV d'entrée
        file_path = f'/home/aziz/Desktop/fatigue_detection/datasets/archive/fatigueset/{participant_id}/{session_id}/wrist_acc.csv'
        
        # Vérifier si le fichier existe
        if not os.path.exists(file_path):
            print(f"Fichier non trouvé : {file_path}")
            continue
            
        print(f"Traitement : Participant {participant_id}, Session {session_id}")
        
        try:
            # Lire le fichier CSV
            df = pd.read_csv(file_path)
            
            # Convertir la colonne 'timestamp' en datetime (en supposant l'unité en ms)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculer le temps de début
            start_time = df['timestamp'].min()
            
            # Convertir la colonne 'timestamp' en secondes relatives
            df['seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
            
            # Calculer la norme L^2 des accéléromètres
            df['norm_L2'] = (df['ax']**2 + df['ay']**2 + df['az']**2)**0.5
            
            # Créer un graphique
            plt.figure(figsize=(10, 6))
            plt.plot(df['seconds'], df['norm_L2'], label='Norme L^2')
            plt.xlabel('Temps (s)')
            plt.ylabel('Norme L^2')
            plt.title(f'Norme L^2 des accéléromètres - P{participant_id} S{session_id}')
            plt.legend()
            plt.grid(True)
            
            # Définir le chemin de sortie
            output_dir = f'/home/aziz/Desktop/fatigue_detection/assets/comparison/acc_Norme_L²/'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'norm_L2_P{participant_id}_S{session_id}.png')
            
            # Enregistrer le graphique
            plt.savefig(output_path, dpi=300)
            plt.close() # Libérer la mémoire
            
            print(f"Graphique enregistré à {output_path}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de P{participant_id} S{session_id} : {e}")

print("Traitement terminé.")
