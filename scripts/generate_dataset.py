import pandas as pd
import numpy as np
import os

def generate_dataset():
    # Chemins des fichiers
    data_path = '/home/aziz/Desktop/fatigue_detection/datasets/data_selected1.csv'
    quest_path = '/home/aziz/Desktop/fatigue_detection/datasets/archive/fatigueset/preliminary_questionnaire.xlsx'
    output_path = '/home/aziz/Desktop/fatigue_detection/datasets/dataset.csv'

    print("Chargement des données...")
    df = pd.read_csv(data_path)
    df_quest = pd.read_excel(quest_path)

    # 1. Labellisation par intervalles
    print("Application de la labellisation par intervalles...")
    df['label'] = "" # Initialiser en tant que chaîne vide
    
    # On itère par participant et session car les intervalles sont spécifiques
    for (p_id, s_id), group in df.groupby(['participant_id', 'session_id']):
        idx = group.index
        
        # Logique pour chaque classe
        for label_name, start_marker, end_marker in [
            ('baseline', 'start_baseline', 'end_baseline'),
            ('activity', 'start_activity', 'end_activity'),
            ('fatigue', 'start_fatigue', 'end_fatigue')
        ]:
            start_indices = group[group['eventMarker'] == start_marker].index
            end_indices = group[group['eventMarker'] == end_marker].index
            
            if not start_indices.empty and not end_indices.empty:
                start_pos = start_indices[0]
                end_pos = end_indices[0]
                df.loc[start_pos:end_pos, 'label'] = label_name

    # Supprimer les lignes sans label (hors intervalles ou marqueurs à supprimer)
    df = df[df['label'] != ""]

    # 2. Nettoyage des signes vitaux selon la fiabilité
    print("Nettoyage des colonnes physiologiques...")
    
    # HR
    # Si hr_confidence < 50 ou is_hr_unreliable == 1 -> NaN
    df.loc[(df['chest_physiology_summary_hr_confidence'] < 50) | 
           (df['chest_physiology_summary_is_hr_unreliable'] == 1), 'chest_physiology_summary_hr'] = np.nan
    
    # BR
    df.loc[df['chest_physiology_summary_is_br_unreliable'] == 1, 'chest_physiology_summary_br'] = np.nan
    
    # HRV
    df.loc[df['chest_physiology_summary_is_hrv_unreliable'] == 1, 'chest_physiology_summary_hrv'] = np.nan

    # 3. Fusion avec le questionnaire
    print("Fusion avec les données statiques...")
    df_quest = df_quest.rename(columns={'ID': 'participant_id'})
    # On ne garde que les participants présents dans df
    df = df.merge(df_quest, on='participant_id', how='left')

    # 4. Interpolation et remplissage des NaNs
    print("Interpolation des valeurs manquantes...")
    # On interpole par participant/session pour ne pas mélanger les données
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df.groupby(['participant_id', 'session_id'])[numeric_cols].transform(lambda x: x.interpolate(method='linear').ffill().bfill())

    # 5. Suppression des colonnes inutiles
    print("Suppression des colonnes inutiles...")
    cols_to_drop = [
        'eventMarker',
        'chest_physiology_summary_hr_confidence',
        'chest_physiology_summary_is_hr_unreliable',
        'chest_physiology_summary_is_br_unreliable',
        'chest_physiology_summary_is_hrv_unreliable',
        'exp_fatigue_measurementNumber',
        'exp_fatigue_physicalFatigueAnswerTime',
        'exp_fatigue_mentalFatigueAnswerTime',
        'exp_fatigue_fatigueSurveySubmissionTime',
        'exp_fatigue_physicalFatigueScore',
        'exp_fatigue_mentalFatigueScore'
    ]
    # S'assurer que les colonnes existent avant de supprimer
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 6. Sauvegarde
    print(f"Sauvegarde du dataset final dans {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Terminé ! Taille du dataset : {df.shape}")

if __name__ == "__main__":
    generate_dataset()
