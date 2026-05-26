import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

CHEMIN_DONNEES = '/media/mohamedaziz-hadjayed/D/aziz_data/fatigue_detection/edge-ai-wearable-fatigue-detection/data/03_processed/dataset_ref.csv'

OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs', 'mfa_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Images sauvegardees dans : {OUTPUT_DIR}")

# ============================================================
# SECTION 1 : CHARGEMENT ET CREATION DES GROUPES MFA (TOUTES FEATURES)
# ============================================================

def charger_donnees(chemin_fichier):
    """Charge le fichier CSV."""
    df = pd.read_csv(chemin_fichier)
    print(f"Donnees chargees : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def creer_groupes_mfa_complet(df):
    """
    Cree les groupes de variables pour la MFA AVEC TOUTES LES FEATURES.
    
    Groupe 1 (Physiologie) : signaux physiologiques quantitatifs
    Groupe 2 (Temporel/Demographique) : timestamp + age
    Groupe 3 (Genre) : gender (qualitatif binaire)
    Groupe 4 (Participant) : participant (identifiant, 12 modalites)
    Groupe 5 (Session) : session (ordinal)
    Groupe 6 (Label) : label (cible binaire)
    """
    # --- GROUPE 1 : PHYSIOLOGIE (quantitatif) ---
    cols_physio = ['acc_x', 'acc_y', 'acc_z', 'eda', 'wrist_hr', 
                   'ibi', 'temp', 'breathing_rpm']
    X_physio = df[cols_physio].copy()
    
    # --- GROUPE 2 : TEMPOREL / DEMOGRAPHIQUE (quantitatif) ---
    cols_temporel = ['timestamp', 'age']
    X_temporel = df[cols_temporel].copy()
    
    # --- GROUPE 3 : GENRE (qualitatif binaire) ---
    gender_dummy = pd.get_dummies(df['gender'], prefix='gender')
    X_genre = gender_dummy.copy()
    
    # --- GROUPE 4 : PARTICIPANT (qualitatif - 12 modalites) ---
    participant_dummy = pd.get_dummies(df['participant'], prefix='participant')
    X_participant = participant_dummy.copy()
    
    # --- GROUPE 5 : SESSION (quantitatif ordinal) ---
    X_session = df[['session']].copy()
    
    # --- GROUPE 6 : LABEL (qualitatif binaire - cible) ---
    label_dummy = pd.get_dummies(df['label'], prefix='label')
    X_label = label_dummy.copy()
    
    groupes = {
        'Physiologie': (X_physio, 'quantitatif'),
        'Temporel_Demo': (X_temporel, 'quantitatif'),
        'Genre': (X_genre, 'qualitatif'),
        'Participant': (X_participant, 'qualitatif'),
        'Session': (X_session, 'quantitatif'),
        'Label': (X_label, 'qualitatif')
    }
    
    print("\n--- Groupes MFA crees (TOUTES FEATURES INTEGREES) ---")
    for nom, (X, type_var) in groupes.items():
        print(f"  {nom:15s} : {X.shape[1]} variables, type={type_var}")
        print(f"    Variables : {list(X.columns)[:5]}{'...' if len(X.columns) > 5 else ''}")
    
    return groupes


# ============================================================
# SECTION 2 : PRETRAITEMENT PAR GROUPE (PCA ou MCA interne)
# ============================================================

def pretraiter_groupe_quantitatif(X, nom_groupe):
    """Pretraite un groupe quantitatif : standardisation + PCA interne."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_comp = min(X.shape[0], X.shape[1])
    pca_interne = PCA(n_components=n_comp)
    X_pca_interne = pca_interne.fit_transform(X_scaled)
    
    premiere_vp = pca_interne.explained_variance_[0]
    poids_groupe = 1.0 / premiere_vp if premiere_vp > 0 else 1.0
    
    print(f"\n  [{nom_groupe}] PCA interne : {n_comp} composantes")
    print(f"    Premiere valeur propre : {premiere_vp:.4f}")
    print(f"    Poids MFA : {poids_groupe:.4f}")
    print(f"    Variance expliquee (1ere comp) : {pca_interne.explained_variance_ratio_[0]*100:.2f}%")
    
    return X_pca_interne, poids_groupe, scaler, pca_interne


def pretraiter_groupe_qualitatif(X, nom_groupe):
    """Pretraite un groupe qualitatif (one-hot) : MCA simplifiee via PCA."""
    X_centered = X - X.mean(axis=0)
    
    n_comp = min(X.shape[0], X.shape[1] - 1) if X.shape[1] > 1 else 1
    pca_interne = PCA(n_components=n_comp)
    X_pca_interne = pca_interne.fit_transform(X_centered)
    
    premiere_vp = pca_interne.explained_variance_[0]
    poids_groupe = 1.0 / premiere_vp if premiere_vp > 0 else 1.0
    
    print(f"\n  [{nom_groupe}] MCA interne (approximation PCA) : {n_comp} composantes")
    print(f"    Premiere valeur propre : {premiere_vp:.4f}")
    print(f"    Poids MFA : {poids_groupe:.4f}")
    
    return X_pca_interne, poids_groupe, None, pca_interne


def pretraiter_tous_groupes(groupes):
    """Applique le pretraitement adapte a chaque groupe."""
    resultats = {}
    
    print("\n" + "=" * 60)
    print("PRETRAITEMENT INTERNE PAR GROUPE")
    print("=" * 60)
    
    for nom, (X, type_var) in groupes.items():
        if type_var == 'quantitatif':
            X_trans, poids, scaler, pca_int = pretraiter_groupe_quantitatif(X, nom)
        else:
            X_trans, poids, scaler, pca_int = pretraiter_groupe_qualitatif(X, nom)
        
        resultats[nom] = {
            'X_original': X,
            'X_transforme': X_trans,
            'poids': poids,
            'scaler': scaler,
            'pca_interne': pca_int,
            'type': type_var
        }
    
    return resultats


# ============================================================
# SECTION 3 : CONSTRUCTION DE LA MATRICE MFA
# ============================================================

def construire_matrice_mfa(resultats_pretraitement):
    """Concatene les groupes pretraites et ponderes."""
    matrices_ponderees = []
    noms_colonnes = []
    poids_par_colonne = []
    infos_groupes = {}
    
    idx_debut = 0
    
    for nom, res in resultats_pretraitement.items():
        X_trans = res['X_transforme']
        poids = res['poids']
        n_comp = X_trans.shape[1]
        
        X_pondere = X_trans * np.sqrt(poids)
        matrices_ponderees.append(X_pondere)
        
        cols_groupe = [f"{nom}_Dim{i+1}" for i in range(n_comp)]
        noms_colonnes.extend(cols_groupe)
        poids_par_colonne.extend([poids] * n_comp)
        
        infos_groupes[nom] = {
            'idx_debut': idx_debut,
            'idx_fin': idx_debut + n_comp,
            'n_comp': n_comp,
            'poids': poids,
            'type': res['type']
        }
        
        idx_debut += n_comp
    
    X_mfa = np.hstack(matrices_ponderees)
    
    print("\n" + "=" * 60)
    print("CONSTRUCTION MATRICE MFA")
    print("=" * 60)
    print(f"Dimensions finales de la matrice MFA : {X_mfa.shape}")
    print(f"Nombre total de dimensions : {len(noms_colonnes)}")
    print(f"Repartition :")
    for nom, info in infos_groupes.items():
        print(f"  {nom:15s} : colonnes {info['idx_debut']} a {info['idx_fin']-1} ({info['n_comp']} dims)")
    
    return X_mfa, noms_colonnes, np.array(poids_par_colonne), infos_groupes


# ============================================================
# SECTION 4 : PCA GLOBALE (MFA)
# ============================================================

def appliquer_mfa_globale(X_mfa, n_comp_mfa=None):
    """Applique la PCA globale sur la matrice MFA ponderee."""
    if n_comp_mfa is None:
        n_comp_mfa = min(X_mfa.shape[0], X_mfa.shape[1])
    
    pca_mfa = PCA(n_components=n_comp_mfa)
    X_mfa_proj = pca_mfa.fit_transform(X_mfa)
    
    print("\n" + "=" * 60)
    print("MFA GLOBALE - RESULTATS")
    print("=" * 60)
    print(f"Composantes MFA calculees : {pca_mfa.n_components_}")
    print(f"Variance expliquee par composante :")
    for i, ratio in enumerate(pca_mfa.explained_variance_ratio_):
        cumul = np.sum(pca_mfa.explained_variance_ratio_[:i+1])
        print(f"  MFA{i+1} : {ratio*100:.2f}%  (cumule : {cumul*100:.2f}%)")
    
    return pca_mfa, X_mfa_proj, pca_mfa.explained_variance_


# ============================================================
# SECTION 5 : ANALYSE DES CONTRIBUTIONS PAR GROUPE
# ============================================================

def analyser_contributions_groupes(pca_mfa, infos_groupes, noms_colonnes):
    """Analyse quels groupes contribuent le plus a chaque composante MFA."""
    n_comp = pca_mfa.n_components_
    contributions = pd.DataFrame(index=[f"MFA{i+1}" for i in range(n_comp)])
    
    loadings = pca_mfa.components_
    
    for nom_groupe, info in infos_groupes.items():
        debut = info['idx_debut']
        fin = info['idx_fin']
        
        contrib_groupe = np.sum(loadings[:, debut:fin] ** 2, axis=1)
        contributions[nom_groupe] = contrib_groupe * 100
    
    contributions = contributions.div(contributions.sum(axis=1), axis=0) * 100
    
    print("\n" + "=" * 60)
    print("CONTRIBUTIONS DES GROUPES PAR COMPOSANTE MFA (%)")
    print("=" * 60)
    print(contributions.round(2).to_string())
    
    return contributions


def analyser_loadings_variables(pca_mfa, noms_colonnes, infos_groupes, n_top=5):
    """Identifie les variables les plus influentes pour chaque composante."""
    loadings = pca_mfa.components_
    n_comp = pca_mfa.n_components_
    
    resultats = {}
    
    print("\n" + "=" * 60)
    print(f"TOP {n_top} VARIABLES MFA PAR COMPOSANTE")
    print("=" * 60)
    
    for i in range(min(n_comp, 5)):
        loadings_comp = np.abs(loadings[i])
        top_idx = np.argsort(loadings_comp)[::-1][:n_top]
        
        print(f"\n--- MFA{i+1} (variance : {pca_mfa.explained_variance_ratio_[i]*100:.2f}%) ---")
        
        top_vars = []
        for idx in top_idx:
            var_name = noms_colonnes[idx]
            loading_val = loadings[i, idx]
            
            groupe_origine = "Inconnu"
            for nom, info in infos_groupes.items():
                if info['idx_debut'] <= idx < info['idx_fin']:
                    groupe_origine = nom
                    break
            
            print(f"  {var_name:30s} : loading = {loading_val:+.4f}  (groupe: {groupe_origine})")
            top_vars.append((var_name, loading_val, groupe_origine))
        
        resultats[f"MFA{i+1}"] = top_vars
    
    return resultats


# ============================================================
# SECTION 6 : VISUALISATIONS
# ============================================================

def sauvegarder_figure(fig, nom_fichier):
    """Sauvegarde une figure dans le repertoire de sortie."""
    chemin_complet = os.path.join(OUTPUT_DIR, nom_fichier)
    fig.savefig(chemin_complet, dpi=150, bbox_inches='tight')
    print(f"  Image sauvegardee : {chemin_complet}")
    plt.close(fig)
    return chemin_complet


def visualiser_scree_plot_mfa(pca_mfa, seuil=0.9):
    """Scree plot MFA."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_comp = pca_mfa.n_components_
    x_pos = np.arange(1, n_comp + 1)
    
    axes[0].bar(x_pos, pca_mfa.explained_variance_, color='darkgreen', alpha=0.7, edgecolor='black')
    axes[0].plot(x_pos, pca_mfa.explained_variance_, 'ro-', markersize=8)
    axes[0].set_xlabel('Composante MFA', fontsize=12)
    axes[0].set_ylabel('Valeur Propre', fontsize=12)
    axes[0].set_title('Scree Plot - MFA Globale', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].grid(axis='y', alpha=0.3)
    
    var_cum = np.cumsum(pca_mfa.explained_variance_ratio_)
    axes[1].bar(x_pos, pca_mfa.explained_variance_ratio_ * 100, color='lightgreen', 
                alpha=0.7, edgecolor='black', label='Par composante')
    axes[1].plot(x_pos, var_cum * 100, 'go-', markersize=8, linewidth=2, label='Cumulee')
    axes[1].axhline(y=seuil*100, color='red', linestyle='--', linewidth=2, label=f'Seuil {int(seuil*100)}%')
    axes[1].set_xlabel('Composante MFA', fontsize=12)
    axes[1].set_ylabel('Variance (%)', fontsize=12)
    axes[1].set_title('Variance Expliquee - MFA', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    sauvegarder_figure(fig, '01_mfa_scree_plot.png')
    print("\nScree plot MFA sauvegarde.")


def visualiser_contributions_groupes(contributions_df):
    """Barres empilees des contributions par groupe."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    comp_names = contributions_df.index
    group_names = contributions_df.columns
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#BC4B51', '#9B5DE5']
    
    bottom = np.zeros(len(comp_names))
    
    for i, groupe in enumerate(group_names):
        ax.bar(comp_names, contributions_df[groupe], bottom=bottom, 
               label=groupe, color=colors[i % len(colors)], alpha=0.85, edgecolor='black', linewidth=0.5)
        bottom += contributions_df[groupe]
    
    ax.set_ylabel('Contribution (%)', fontsize=12)
    ax.set_xlabel('Composante MFA', fontsize=12)
    ax.set_title('Contributions des Groupes par Composante MFA\n(Toutes features integrees)', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Groupe', loc='upper right', ncol=2)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    sauvegarder_figure(fig, '02_mfa_contributions_groupes.png')
    print("\nGraphique des contributions sauvegarde.")


def visualiser_projection_mfa(X_mfa_proj, y=None, pc_x=1, pc_y=2):
    """Projection des individus coloree par label."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if y is not None:
        classes = np.unique(y)
        colors = ['#2E86AB', '#F24236', '#2EC4B6', '#9B5DE5']
        
        for i, classe in enumerate(classes):
            mask = y == classe
            ax.scatter(X_mfa_proj[mask, pc_x-1], X_mfa_proj[mask, pc_y-1], 
                       c=colors[i % len(colors)], label=f'Label {classe}', 
                       alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        ax.legend(title='Label', loc='best')
    else:
        ax.scatter(X_mfa_proj[:, pc_x-1], X_mfa_proj[:, pc_y-1], 
                   c='darkgreen', alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'MFA{pc_x}', fontsize=12)
    ax.set_ylabel(f'MFA{pc_y}', fontsize=12)
    ax.set_title(f'Projection MFA - MFA{pc_x} vs MFA{pc_y}\n(Toutes features)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    sauvegarder_figure(fig, '03_mfa_projection_individus.png')
    print(f"\nProjection MFA (MFA{pc_x} vs MFA{pc_y}) sauvegardee.")


def visualiser_projection_par_meta(X_mfa_proj, meta_var, titre, nom_fichier, pc_x=1, pc_y=2):
    """Projection coloree par variable meta."""
    fig, ax = plt.subplots(figsize=(10, 8))
    valeurs_uniques = np.unique(meta_var)
    n_vals = len(valeurs_uniques)
    
    if n_vals <= 3:
        colors = ['#2E86AB', '#F24236', '#2EC4B6']
    elif n_vals <= 12:
        colors = plt.cm.tab10(np.linspace(0, 1, n_vals))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_vals))
    
    for i, val in enumerate(valeurs_uniques):
        mask = meta_var == val
        ax.scatter(X_mfa_proj[mask, pc_x-1], X_mfa_proj[mask, pc_y-1], 
                   c=[colors[i]], label=f'{val}', alpha=0.6, s=50, edgecolors='black', linewidth=0.3)
    
    ax.set_xlabel(f'MFA{pc_x}', fontsize=12)
    ax.set_ylabel(f'MFA{pc_y}', fontsize=12)
    ax.set_title(f'{titre} - MFA{pc_x} vs MFA{pc_y}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if n_vals <= 6:
        ax.legend(title=titre.split()[-1], loc='best', fontsize=9)
    else:
        ax.legend(title=titre.split()[-1], loc='best', fontsize=7, ncol=2)
    
    plt.tight_layout()
    sauvegarder_figure(fig, nom_fichier)
    print(f"\nProjection par {titre} sauvegardee.")


# ============================================================
# SECTION 7 : PIPELINE MFA COMPLET (TOUTES FEATURES)
# ============================================================

def pipeline_mfa_complet(chemin_fichier, n_comp_mfa=None):
    """
    Pipeline complet de la MFA avec TOUTES les features integrees.
    """
    print("=" * 70)
    print("    PIPELINE COMPLET - ANALYSE MFA (TOUTES FEATURES)")
    print("=" * 70)
    print("\nGroupes integres :")
    print("  [1] Physiologie     : acc_x, acc_y, acc_z, eda, wrist_hr, ibi, temp, breathing_rpm")
    print("  [2] Temporel/Demo   : timestamp, age")
    print("  [3] Genre           : gender (one-hot)")
    print("  [4] Participant     : participant (one-hot, 12 modalites)")
    print("  [5] Session         : session")
    print("  [6] Label           : label (one-hot, cible)")
    
    # --- ETAPE 1 : Chargement ---
    print("\n" + "-" * 50)
    print("ETAPE 1 : CHARGEMENT")
    print("-" * 50)
    df = charger_donnees(chemin_fichier)
    
    # --- ETAPE 2 : Creation des groupes ---
    print("\n" + "-" * 50)
    print("ETAPE 2 : CREATION DES GROUPES MFA (COMPLET)")
    print("-" * 50)
    groupes = creer_groupes_mfa_complet(df)
    
    # --- ETAPE 3 : Pretraitement interne ---
    print("\n" + "-" * 50)
    print("ETAPE 3 : PRETRAITEMENT INTERNE (PCA/MCA)")
    print("-" * 50)
    resultats_pretrait = pretraiter_tous_groupes(groupes)
    
    # --- ETAPE 4 : Matrice MFA ---
    print("\n" + "-" * 50)
    print("ETAPE 4 : CONSTRUCTION MATRICE MFA")
    print("-" * 50)
    X_mfa, noms_cols, poids_cols, infos_groupes = construire_matrice_mfa(resultats_pretrait)
    
    # --- ETAPE 5 : MFA globale ---
    print("\n" + "-" * 50)
    print("ETAPE 5 : MFA GLOBALE")
    print("-" * 50)
    pca_mfa, X_mfa_proj, vp_mfa = appliquer_mfa_globale(X_mfa, n_comp_mfa)
    
    # --- ETAPE 6 : Analyse des contributions ---
    print("\n" + "-" * 50)
    print("ETAPE 6 : ANALYSE DES CONTRIBUTIONS")
    print("-" * 50)
    contrib_groupes = analyser_contributions_groupes(pca_mfa, infos_groupes, noms_cols)
    top_vars = analyser_loadings_variables(pca_mfa, noms_cols, infos_groupes, n_top=5)
    
    # --- ETAPE 7 : Visualisations ---
    print("\n" + "-" * 50)
    print("ETAPE 7 : VISUALISATIONS")
    print("-" * 50)
    
    visualiser_scree_plot_mfa(pca_mfa, seuil=0.9)
    visualiser_contributions_groupes(contrib_groupes)
    
    # Projection par label (cible)
    y = df['label'].values if 'label' in df.columns else None
    visualiser_projection_mfa(X_mfa_proj, y=y, pc_x=1, pc_y=2)
    
    # Projections par meta-variables
    if 'participant' in df.columns:
        visualiser_projection_par_meta(X_mfa_proj, df['participant'].values, 
                                       "Par Participant", "05_mfa_par_participant.png")
    if 'session' in df.columns:
        visualiser_projection_par_meta(X_mfa_proj, df['session'].values, 
                                       "Par Session", "06_mfa_par_session.png")
    if 'gender' in df.columns:
        visualiser_projection_par_meta(X_mfa_proj, df['gender'].values, 
                                       "Par Genre", "07_mfa_par_gender.png")
    
    print("\n" + "=" * 70)
    print("    ANALYSE MFA TERMINEE")
    print("=" * 70)
    print(f"\nToutes les images sont dans : {OUTPUT_DIR}")
    
    return {
        'df': df,
        'groupes': groupes,
        'resultats_pretraitement': resultats_pretrait,
        'X_mfa': X_mfa,
        'noms_colonnes_mfa': noms_cols,
        'pca_mfa': pca_mfa,
        'X_mfa_proj': X_mfa_proj,
        'contributions_groupes': contrib_groupes,
        'top_variables': top_vars,
        'infos_groupes': infos_groupes
    }


# ============================================================
# SECTION 8 : EXECUTION
# ============================================================

if __name__ == "__main__":
    if not os.path.exists(CHEMIN_DONNEES):
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"ERREUR : Fichier non trouve !\n"
            f"Chemin : {CHEMIN_DONNEES}\n"
            f"Cwd : {os.getcwd()}\n"
            f"Modifiez CHEMIN_DONNEES ligne 17."
            f"\n{'='*60}"
        )
    
    print(f"Fichier trouve : {CHEMIN_DONNEES}")
    print(f"Taille : {os.path.getsize(CHEMIN_DONNEES):,} octets")
    
    # Lancement avec toutes les features
    resultats_mfa = pipeline_mfa_complet(CHEMIN_DONNEES, n_comp_mfa=None)