#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de comparaison des modèles AI pour Edge AI
Lit metrics.json et les dossiers models_saved/ pour générer un tableau comparatif
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
METRICS_FILE = "metrics.json"           # Fichier de métriques
MODELS_DIR = "models_saved"              # Dossier des modèles sauvegardés
OUTPUT_FILE = "comparaison_modeles.html" # Fichier de sortie

# ============================================================
# 1. CHARGEMENT DES MÉTRIQUES
# ============================================================
def load_metrics(filepath):
    """Charge le fichier metrics.json"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# ============================================================
# 2. CALCUL DES TAILLES DES MODÈLES
# ============================================================
def get_model_size(model_name, models_dir):
    """
    Calcule la taille totale du dossier du modèle en MB
    Cherche dans models_saved/{model_name}/
    """
    model_path = Path(models_dir) / model_name
    
    if not model_path.exists():
        return None
    
    total_size = 0
    file_count = 0
    
    for file_path in model_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
            file_count += 1
    
    size_mb = total_size / (1024 * 1024)
    return {
        'size_mb': round(size_mb, 2),
        'size_kb': round(total_size / 1024, 2),
        'file_count': file_count
    }

# ============================================================
# 3. EXTRACTION DES MÉTRIQUES GLOBALES PAR MODÈLE
# ============================================================
def extract_model_metrics(model_name, model_data):
    """Extrait les métriques globales d'un modèle"""
    metrics = {
        'Modele': model_name,
        'F1_Macro_Moyenne': model_data.get('mean_f1', np.nan),
        'F1_Macro_Std': model_data.get('std_f1', np.nan),
        'Bal_Acc_Moyenne': model_data.get('mean_bal_acc', np.nan),
        'Bal_Acc_Std': model_data.get('std_bal_acc', np.nan),
    }
    
    # Nombre de folds (cross-validation)
    folds = model_data.get('folds', [])
    metrics['Nb_Folds'] = len(folds)
    
    # Calcul du min/max F1 sur les folds
    if folds:
        f1_scores = [f.get('F1_Macro', np.nan) for f in folds]
        metrics['F1_Min'] = round(min(f1_scores), 3)
        metrics['F1_Max'] = round(max(f1_scores), 3)
    
    # Paramètres clés
    params = model_data.get('params', {})
    metrics['Params'] = str(params) if params else 'N/A'
    
    return metrics

# ============================================================
# 4. SCORE EDGE AI (pondération pour edge)
# ============================================================
def calculate_edge_score(row, weight_f1=0.5, weight_size=0.3, weight_stability=0.2):
    """
    Calcule un score composite pour l'Edge AI :
    - Performance F1 (50%)
    - Taille modèle inverse (30%) - plus petit = meilleur
    - Stabilité (std faible) (20%)
    """
    f1 = row['F1_Macro_Moyenne']
    size_mb = row['Taille_MB'] if pd.notna(row['Taille_MB']) else 100  # valeur par défaut
    
    # Normaliser la taille (plus petit = meilleur score)
    max_size = 500  # MB, valeur de référence
    size_score = max(0, 1 - (size_mb / max_size))
    
    # Stabilité (inverse de l'écart-type)
    std_f1 = row['F1_Macro_Std']
    stability_score = max(0, 1 - std_f1)
    
    edge_score = (weight_f1 * f1 + 
                  weight_size * size_score + 
                  weight_stability * stability_score)
    
    return round(edge_score, 3)

# ============================================================
# 5. GÉNÉRATION DU TABLEAU COMPARATIF
# ============================================================
def generate_comparison_table(metrics_data, models_dir):
    """Génère le DataFrame comparatif complet"""
    
    comparison_data = []
    
    for model_name, model_info in metrics_data.items():
        # Métriques de performance
        metrics = extract_model_metrics(model_name, model_info)
        
        # Taille du modèle
        size_info = get_model_size(model_name, models_dir)
        if size_info:
            metrics['Taille_MB'] = size_info['size_mb']
            metrics['Taille_KB'] = size_info['size_kb']
            metrics['Nb_Fichiers'] = size_info['file_count']
        else:
            metrics['Taille_MB'] = np.nan
            metrics['Taille_KB'] = np.nan
            metrics['Nb_Fichiers'] = 0
        
        comparison_data.append(metrics)
    
    # Créer le DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Calculer le score Edge AI
    df['Score_Edge_AI'] = df.apply(calculate_edge_score, axis=1)
    
    # Trier par score Edge AI décroissant
    df = df.sort_values('Score_Edge_AI', ascending=False).reset_index(drop=True)
    
    # Ajouter le rang
    df['Rang'] = range(1, len(df) + 1)
    
    # Réorganiser les colonnes
    cols_order = [
        'Rang', 'Modele', 
        'F1_Macro_Moyenne', 'F1_Macro_Std', 
        'Bal_Acc_Moyenne', 'Bal_Acc_Std',
        'F1_Min', 'F1_Max', 'Nb_Folds',
        'Taille_MB', 'Taille_KB', 'Nb_Fichiers',
        'Score_Edge_AI', 'Params'
    ]
    
    df = df[[c for c in cols_order if c in df.columns]]
    
    return df

# ============================================================
# 6. FONCTIONS DE COULEUR POUR HTML (sans jinja2)
# ============================================================
def get_color_class(value, metric_type):
    """
    Retourne une classe CSS selon la valeur et le type de métrique
    """
    if pd.isna(value):
        return ""
    
    if metric_type == 'f1':
        if value >= 0.8:
            return "excellent"
        elif value >= 0.6:
            return "good"
        else:
            return "poor"
    
    elif metric_type == 'size':
        if value <= 10:
            return "excellent"
        elif value <= 50:
            return "good"
        else:
            return "poor"
    
    elif metric_type == 'score':
        if value >= 0.7:
            return "excellent"
        elif value >= 0.5:
            return "good"
        else:
            return "poor"
    
    return ""

# ============================================================
# 7. EXPORT EN HTML (sans dépendance jinja2)
# ============================================================
def export_to_html(df, output_file):
    """Exporte le DataFrame en HTML avec mise en forme manuelle"""
    
    # CSS intégré
    css = """
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f6fa;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .summary {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        th {
            background-color: #34495e;
            color: white;
            font-weight: bold;
            text-align: center;
            padding: 12px;
            font-size: 14px;
        }
        td {
            padding: 10px;
            text-align: center;
            border-bottom: 1px solid #ecf0f1;
            font-size: 13px;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #e8f4f8;
        }
        .excellent {
            background-color: #2ecc71 !important;
            color: white !important;
            font-weight: bold;
        }
        .good {
            background-color: #f39c12 !important;
            color: white !important;
        }
        .poor {
            background-color: #e74c3c !important;
            color: white !important;
        }
        .legend {
            margin-top: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            font-size: 13px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .legend-item {
            display: inline-block;
            margin: 5px 15px 5px 0;
        }
        .color-box {
            display: inline-block;
            width: 20px;
            height: 20px;
            vertical-align: middle;
            margin-right: 5px;
            border-radius: 3px;
        }
        .params-cell {
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 11px;
            text-align: left;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }
    </style>
    """
    
    # Construire le tableau HTML ligne par ligne
    rows_html = []
    
    # En-têtes
    headers = list(df.columns)
    header_html = "<tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>"
    
    # Données
    for idx, row in df.iterrows():
        row_html = "<tr>"
        for col in headers:
            val = row[col]
            
            # Formater la valeur
            if pd.isna(val):
                display_val = "N/A"
            elif isinstance(val, float):
                if col in ['F1_Macro_Moyenne', 'F1_Macro_Std', 'Bal_Acc_Moyenne', 
                           'Bal_Acc_Std', 'F1_Min', 'F1_Max', 'Score_Edge_AI']:
                    display_val = f"{val:.3f}"
                elif col in ['Taille_MB', 'Taille_KB']:
                    display_val = f"{val:.2f}"
                else:
                    display_val = f"{val:.3f}"
            else:
                display_val = str(val)
            
            # Appliquer les classes de couleur
            css_class = ""
            if col in ['F1_Macro_Moyenne', 'Bal_Acc_Moyenne']:
                css_class = get_color_class(val, 'f1')
            elif col == 'Taille_MB':
                css_class = get_color_class(val, 'size')
            elif col == 'Score_Edge_AI':
                css_class = get_color_class(val, 'score')
            
            # Classe spéciale pour les params
            cell_class = f'class="{css_class}"' if css_class else ''
            if col == 'Params':
                cell_class = 'class="params-cell"'
            
            row_html += f'<td {cell_class}>{display_val}</td>'
        
        row_html += "</tr>"
        rows_html.append(row_html)
    
    # Assembler le tableau
    table_html = f"<table>\n<thead>\n{header_html}\n</thead>\n<tbody>\n" + "\n".join(rows_html) + "\n</tbody>\n</table>"
    
    # Contenu HTML complet
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Comparaison Modèles AI - Edge AI</title>
    {css}
</head>
<body>
    <h1>📊 Tableau Comparatif des Modèles AI - Edge AI</h1>
    
    <div class="summary">
        <h3>📋 Résumé de l'analyse</h3>
        <p><strong>Nombre de modèles comparés:</strong> {len(df)}</p>
        <p><strong>Meilleur modèle (Score Edge AI):</strong> {df.iloc[0]['Modele']} (Score: {df.iloc[0]['Score_Edge_AI']:.3f})</p>
        <p><strong>Meilleur F1 Macro:</strong> {df.loc[df['F1_Macro_Moyenne'].idxmax(), 'Modele']} ({df['F1_Macro_Moyenne'].max():.3f})</p>
        <p><strong>Modèle le plus léger:</strong> {df.loc[df['Taille_MB'].idxmin(), 'Modele']} ({df['Taille_MB'].min():.2f} MB)</p>
    </div>
    
    {table_html}
    
    <div class="legend">
        <h4>🎨 Légende des couleurs</h4>
        <div class="legend-item">
            <span class="color-box" style="background-color:#2ecc71;"></span> Excellent (F1 ≥ 0.8 / Taille ≤ 10 MB / Score ≥ 0.7)
        </div>
        <div class="legend-item">
            <span class="color-box" style="background-color:#f39c12;"></span> Bon (F1 0.6-0.8 / Taille 10-50 MB / Score 0.5-0.7)
        </div>
        <div class="legend-item">
            <span class="color-box" style="background-color:#e74c3c;"></span> À améliorer (F1 < 0.6 / Taille > 50 MB / Score < 0.5)
        </div>
    </div>
    
    <div class="footer">
        Généré automatiquement le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Tableau HTML exporté: {output_file}")

# ============================================================
# 8. FONCTION PRINCIPALE
# ============================================================
def main():
    print("=" * 60)
    print("COMPARATEUR DE MODÈLES AI - EDGE AI")
    print("=" * 60)
    
    # Vérifier l'existence du fichier metrics.json
    if not os.path.exists(METRICS_FILE):
        print(f"❌ Erreur: Fichier {METRICS_FILE} non trouvé!")
        return
    
    # Charger les métriques
    print(f"📂 Chargement de {METRICS_FILE}...")
    metrics_data = load_metrics(METRICS_FILE)
    print(f"   → {len(metrics_data)} modèles trouvés")
    
    # Vérifier le dossier models_saved
    if not os.path.exists(MODELS_DIR):
        print(f"⚠️  Attention: Dossier {MODELS_DIR} non trouvé. Les tailles seront indisponibles.")
    
    # Générer le tableau comparatif
    print("\n📊 Génération du tableau comparatif...")
    df_comparison = generate_comparison_table(metrics_data, MODELS_DIR)
    
    # Afficher dans la console
    print("\n" + "=" * 60)
    print("TABLEAU COMPARATIF")
    print("=" * 60)
    
    # Affichage formaté pour la console
    display_df = df_comparison.copy()
    for col in ['F1_Macro_Moyenne', 'F1_Macro_Std', 'Bal_Acc_Moyenne', 
                'Bal_Acc_Std', 'F1_Min', 'F1_Max', 'Score_Edge_AI']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    for col in ['Taille_MB', 'Taille_KB']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    print(display_df.to_string())
    
    # Export CSV
    csv_file = OUTPUT_FILE.replace('.html', '.csv')
    df_comparison.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\n✅ Export CSV: {csv_file}")
    
    # Export HTML
    export_to_html(df_comparison, OUTPUT_FILE)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 60)
    print(f"\n📁 Fichiers générés:")
    print(f"   • {csv_file}")
    print(f"   • {OUTPUT_FILE}")

# ============================================================
# EXÉCUTION
# ============================================================
if __name__ == "__main__":
    main()