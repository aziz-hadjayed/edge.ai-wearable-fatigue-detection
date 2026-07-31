import os
# pyrefly: ignore [missing-import]
import torch
import shutil
import glob
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION GLOBALE
# ============================================

DATASET_PATH = "data/05_vision_splited/dataset"
MODEL_NAME = "yolov8n-cls"
OUTPUT_MODEL_PATH = "models_saved/YOLOV8n/yolov8_classify.onnx"
OUTPUT_CURVES_PATH = "training_curves/YOLOV8n"
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# MAPPING DES CLASSES
# ============================================
# YOLO classe les classes par ordre alphabetique
# Donc: fatigue (0) et non_fatigue (1) dans l'ordre YOLO
# On inverse pour avoir: non_fatigue (0) et fatigue (1)
CLASS_MAP = {0: 1, 1: 0}

# ============================================
# FONCTION: TROUVER LE DOSSIER D'ENTRAINEMENT
# ============================================

def find_training_folder():
    """
    Trouve automatiquement le dossier d'entrainement le plus recent.
    YOLO cree des sous-dossiers qui peuvent etre imbriques.
    
    Returns:
        str: Chemin vers le dossier d'entrainement ou None
    """
    
    # Recherche recursive de tous les dossiers 'exp' dans runs/classify
    search_pattern = "runs/classify/**/exp"
    matching_dirs = glob.glob(search_pattern, recursive=True)
    
    if not matching_dirs:
        print("Aucun dossier d'entrainement trouve")
        return None
    
    # Trie par date de modification (le plus recent en premier)
    matching_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_exp = matching_dirs[0]
    
    print(f"Dossier d'entrainement trouve: {latest_exp}")
    return latest_exp


# ============================================
# FONCTION 1: ENTRAINEMENT DU MODELE
# ============================================

def train_yolov8():
    """
    Entraine le modele YOLOv8n pour la classification binaire.
    
    Returns:
        tuple: (model, results, exp_path) - Le modele, les resultats, et le chemin du dossier exp
    """
    
    # Chargement du modele pre-entraine
    model = YOLO(f"{MODEL_NAME}.pt")
    
    # Lancement de l'entrainement avec les hyperparametres configures
    results = model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        optimizer='Adam',
        lr0=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        seed=42,
        resume=False,
        exist_ok=True,
        project=f"runs/classify/runs/train/{MODEL_NAME}",
        name="exp",
        save=True,
        save_period=-1,
        plots=False
    )
    
    # Recuperation du chemin du dossier d'entrainement
    exp_path = find_training_folder()
    
    return model, results, exp_path


# ============================================
# FONCTION 2: CHARGEMENT DE L'HISTORIQUE
# ============================================

def load_training_history(exp_path):
    """
    Charge l'historique d'entrainement depuis le fichier CSV.
    
    Args:
        exp_path: Chemin du dossier d'entrainement (trouve automatiquement)
        
    Returns:
        DataFrame ou None: Les donnees d'entrainement ou None si non trouve
    """
    
    if exp_path is None:
        print("Chemin d'entrainement non disponible")
        return None
    
    # Le CSV est dans le dossier exp
    csv_path = os.path.join(exp_path, "results.csv")
    
    if os.path.exists(csv_path):
        print(f"Historique d'entrainement trouve: {csv_path}")
        df = pd.read_csv(csv_path)
        return df
    else:
        print(f"Fichier CSV non trouve: {csv_path}")
        # Liste les fichiers disponibles dans le dossier pour debug
        print("Fichiers disponibles dans le dossier:")
        for f in os.listdir(exp_path):
            print(f"  - {f}")
        return None


# ============================================
# FONCTION 3: EVALUATION DU MODELE
# ============================================

def evaluate_model(model):
    """
    Evalue le modele sur l'ensemble de test et calcule les metriques.
    
    Args:
        model: Le modele YOLO entraine
        
    Returns:
        dict ou None: Dictionnaire des metriques ou None si erreur
    """
    
    # Chemin vers les donnees de test
    test_path = os.path.join(DATASET_PATH, 'test')
    test_data = []
    test_labels = []
    
    # Parcours des deux classes dans le dossier test
    for class_name in ['non_fatigue', 'fatigue']:
        class_path = os.path.join(test_path, class_name)
        class_label = 0 if class_name == 'non_fatigue' else 1
        
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    test_data.append(img_path)
                    test_labels.append(class_label)
    
    if not test_data:
        print("Aucune donnee de test trouvee!")
        return None
    
    predictions = []
    confidences = []
    
    for img_path in test_data:
        results = model(img_path)
        
        if len(results) > 0 and results[0].probs is not None:
            probs = results[0].probs.data.cpu().numpy()
            yolo_pred = np.argmax(probs)
            confidence = np.max(probs)
            pred_class = CLASS_MAP.get(yolo_pred, yolo_pred)
            predictions.append(pred_class)
            confidences.append(confidence)
        else:
            predictions.append(0)
            confidences.append(0.0)
    
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='binary', zero_division=0)
    recall = recall_score(test_labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='binary', zero_division=0)
    cm = confusion_matrix(test_labels, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': predictions,
        'test_labels': test_labels,
        'confidences': confidences
    }
    
    print("\n=== Metriques d'Evaluation ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision:              {precision:.4f}")
    print(f"Rappel (Recall):        {recall:.4f}")
    print(f"Score F1:               {f1:.4f}")
    
    return metrics


# ============================================
# FONCTION 4: VISUALISATION DES RESULTATS
# ============================================
def evaluate_dataset_accuracy(model, split='val'):
    """
    Evalue le modele sur un split du dataset et retourne son accuracy.
    """
    
    split_path = os.path.join(DATASET_PATH, split)
    split_data = []
    split_labels = []
    
    for class_name in ['non_fatigue', 'fatigue']:
        class_path = os.path.join(split_path, class_name)
        class_label = 0 if class_name == 'non_fatigue' else 1
        
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    split_data.append(img_path)
                    split_labels.append(class_label)
    
    if not split_data:
        print(f"Aucune donnee trouvee pour le split: {split}")
        return None
    
    predictions = []
    
    for img_path in split_data:
        results = model(img_path)
        if len(results) > 0 and results[0].probs is not None:
            probs = results[0].probs.data.cpu().numpy()
            yolo_pred = np.argmax(probs)
            pred_class = CLASS_MAP.get(yolo_pred, yolo_pred)
            predictions.append(pred_class)
        else:
            predictions.append(0)
    
    return accuracy_score(split_labels, predictions)


def evaluate_validation_set(model):
    """
    Evalue le modele sur l'ensemble de validation.
    """
    return evaluate_dataset_accuracy(model, split='val')


def plot_training_curves_and_confusion_matrix(history_df, metrics, model=None):
    """
    Genere un graphique avec les courbes d'entrainement et la matrice de confusion.
    
    Args:
        history_df: DataFrame avec l'historique d'entrainement
        metrics: Dictionnaire des metriques d'evaluation
    """
    
    if history_df is None:
        print("Aucun historique d'entrainement disponible")
        return
    
    print("Generation des courbes d'entrainement...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # --- Graphique 1: Courbe de perte ---
    ax1 = plt.subplot(2, 3, 1)
    epochs = history_df['epoch'].values + 1
    
    if 'train/loss' in history_df.columns:
        ax1.plot(epochs, history_df['train/loss'], 
                label='Perte Entrainement', color='blue', linewidth=2)
    if 'val/loss' in history_df.columns:
        ax1.plot(epochs, history_df['val/loss'], 
                label='Perte Validation', color='red', linewidth=2)
        
        best_idx = history_df['val/loss'].idxmin()
        best_epoch = epochs[best_idx]
        best_loss = history_df['val/loss'].min()
        ax1.scatter(best_epoch, best_loss, color='red', s=100, zorder=5, marker='*')
        ax1.annotate(f'Meilleur: {best_loss:.4f}', 
                    xy=(best_epoch, best_loss),
                    xytext=(best_epoch + 2, best_loss + 0.02),
                    fontsize=10, color='red')
    
    ax1.set_title('Courbes de Perte', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoque')
    ax1.set_ylabel('Perte')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Graphique 2: Courbe d'Accuracy ---
    ax2 = plt.subplot(2, 3, 2)
    
    if 'metrics/accuracy_top1' in history_df.columns:
        ax2.plot(epochs, history_df['metrics/accuracy_top1'], 
                label='Accuracy Validation', color='orange', linewidth=2)
        
        best_idx = history_df['metrics/accuracy_top1'].idxmax()
        best_epoch = epochs[best_idx]
        best_acc = history_df['metrics/accuracy_top1'].max()
        ax2.scatter(best_epoch, best_acc, color='orange', s=100, zorder=5, marker='*')
        ax2.annotate(f'Meilleur: {best_acc:.4f}', 
                    xy=(best_epoch, best_acc),
                    xytext=(best_epoch + 2, best_acc - 0.02),
                    fontsize=10, color='orange')
    
    train_acc_cols = [col for col in history_df.columns
                     if 'train' in col.lower() and 'acc' in col.lower()]
    if train_acc_cols:
        ax2.plot(epochs, history_df[train_acc_cols[0]],
                label='Accuracy Entrainement', color='green', linewidth=2)
    
    if model is not None:
        print("Calcul de l'accuracy d'entrainement...")
        train_acc = evaluate_dataset_accuracy(model, split='train')
        if train_acc is not None:
            ax2.axhline(y=train_acc, color='green', linestyle='--',
                       linewidth=2, label=f'Accuracy Entrainement Finale: {train_acc:.4f}')
            print(f"Accuracy d'entrainement: {train_acc:.4f}")
        
        print("Calcul de l'accuracy de validation...")
        val_acc = evaluate_validation_set(model)
        if val_acc is not None:
            ax2.axhline(y=val_acc, color='red', linestyle='--', 
                       linewidth=2, label=f'Accuracy Validation Finale: {val_acc:.4f}')
            print(f"Accuracy de validation: {val_acc:.4f}")
    
    ax2.set_title('Courbes d Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoque')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # --- Graphique 3: Taux d'apprentissage ---
    ax3 = plt.subplot(2, 3, 3)
    
    if 'lr/pg0' in history_df.columns:
        ax3.plot(epochs, history_df['lr/pg0'], 
                label='Taux d Apprentissage', color='purple', linewidth=2)
    ax3.set_title('Evolution du Taux d Apprentissage', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoque')
    ax3.set_ylabel('Taux d Apprentissage')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Graphique 4: Metriques de performance ---
    ax4 = plt.subplot(2, 3, 4)
    
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'accuracy']
    values = [metrics[m] for m in metrics_to_plot if m in metrics]
    labels = [m.replace('_', ' ').title() for m in metrics_to_plot if m in metrics]
    
    if values and labels:
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
        bars = ax4.bar(labels, values, color=colors[:len(labels)], 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Score', fontsize=11)
        ax4.set_title('Metriques de Performance', fontsize=14, fontweight='bold')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
    
    # --- Graphique 5: Matrice de confusion ---
    ax5 = plt.subplot(2, 3, 5)
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Fatigue', 'Fatigue'],
                   yticklabels=['Non-Fatigue', 'Fatigue'],
                   ax=ax5, 
                   cbar_kws={'label': 'Nombre'},
                   annot_kws={'size': 16, 'weight': 'bold'})
        ax5.set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Predi', fontsize=12)
        ax5.set_ylabel('Reel', fontsize=12)
    
    # --- Graphique 6: Distribution des classes ---
    ax6 = plt.subplot(2, 3, 6)
    if 'test_labels' in metrics:
        labels = ['Non-Fatigue', 'Fatigue']
        counts = [np.sum(np.array(metrics['test_labels']) == 0), 
                  np.sum(np.array(metrics['test_labels']) == 1)]
        
        colors = ['#4CAF50', '#FF5722']
        bars = ax6.bar(labels, counts, color=colors, 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax6.set_title('Echantillons par Classe (Test)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Classe', fontsize=12)
        ax6.set_ylabel('Nombre d Echantillons', fontsize=12)
        ax6.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(count)}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=12)
    
    fig.suptitle('Resultats de Classification YOLOv8n', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    os.makedirs(OUTPUT_CURVES_PATH, exist_ok=True)
    output_path = os.path.join(OUTPUT_CURVES_PATH, 'training_curves_and_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCourbes et metriques sauvegardees: {output_path}")
    plt.close()


# ============================================
# FONCTION 5: EXPORTATION EN ONNX
# ============================================

def save_onnx_model(model, exp_path):
    """
    Exporte le modele au format ONNX et le copie au bon endroit.
    
    Args:
        model: Le modele YOLO entraine
        exp_path: Chemin du dossier d'entrainement
    """
    
    print("Exportation du modele vers ONNX...")
    
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    
    # Exportation au format ONNX
    model.export(
        format='onnx',
        imgsz=IMG_SIZE,
        opset=12,
        dynamic=True,
        simplify=True
    )
    
    # ============================================
    # RECHERCHE DU FICHIER ONNX (methode robuste)
    # ============================================
    
    # 1. D'abord chercher dans le dossier weights du dossier exp
    onnx_found = False
    
    if exp_path:
        weights_dir = os.path.join(exp_path, "weights")
        if os.path.exists(weights_dir):
            # Cherche tous les fichiers .onnx dans weights
            onnx_files = glob.glob(os.path.join(weights_dir, "*.onnx"))
            if onnx_files:
                source_path = onnx_files[0]  # Prend le premier trouve
                print(f"Fichier ONNX trouve dans weights: {source_path}")
                shutil.copy2(source_path, OUTPUT_MODEL_PATH)
                print(f"Modele copie vers: {OUTPUT_MODEL_PATH}")
                onnx_found = True
    
    # 2. Si pas trouve, recherche recursive dans tout le projet
    if not onnx_found:
        print("Recherche recursive du fichier ONNX...")
        search_results = glob.glob("**/*.onnx", recursive=True)
        
        # Exclure le dossier de destination pour eviter de se copier soi-meme
        search_results = [p for p in search_results 
                         if not p.startswith(os.path.dirname(OUTPUT_MODEL_PATH))]
        
        if search_results:
            # Prend le plus recent
            search_results.sort(key=os.path.getmtime, reverse=True)
            source_path = search_results[0]
            print(f"Fichier ONNX trouve: {source_path}")
            shutil.copy2(source_path, OUTPUT_MODEL_PATH)
            print(f"Modele copie vers: {OUTPUT_MODEL_PATH}")
            onnx_found = True
    
    # 3. Si toujours pas trouve, message d'erreur
    if not onnx_found:
        print("ERREUR: Fichier ONNX non trouve apres exportation")
        print("Recherche dans le repertoire courant:")
        for root, dirs, files in os.walk("."):
            for f in files:
                if f.endswith('.onnx'):
                    print(f"  Trouve: {os.path.join(root, f)}")


# ============================================
# FONCTION PRINCIPALE
# ============================================

def main():
    """
    Fonction principale orchestrant l'entrainement, l'evaluation 
    et la sauvegarde du modele.
    """
    
    print("=" * 60)
    print("ENTRAINEMENT YOLOv8n - Classification Binaire")
    print("=" * 60)
    print(f"Peripherique utilise: {DEVICE}")
    print(f"Mapping des classes:")
    print(f"  Classe YOLO 0 (fatigue)      -> Classe 1")
    print(f"  Classe YOLO 1 (non_fatigue)  -> Classe 0")
    print("=" * 60)
    
    # ============================================
    # ETAPE 1: ENTRAINEMENT (retourne aussi le chemin exp)
    # ============================================
    print("\n[1/4] Entrainement du modele...")
    model, results, exp_path = train_yolov8()
    
    # ============================================
    # ETAPE 2: CHARGEMENT DE L'HISTORIQUE (avec chemin auto)
    # ============================================
    print("\n[2/4] Chargement de l historique d entrainement...")
    history_df = load_training_history(exp_path)
    if history_df is not None:
        print(f"Historique charge: {len(history_df)} epoques")
        print(f"Colonnes disponibles: {list(history_df.columns)}")
    else:
        print("ATTENTION: Impossible de charger l historique")
        print("Les courbes ne seront pas generees")
    
    # ============================================
    # ETAPE 3: EVALUATION
    # ============================================
    print("\n[3/4] Evaluation sur l ensemble de test...")
    metrics = evaluate_model(model)
    
    # ============================================
    # ETAPE 4: VISUALISATION
    # ============================================
    print("\n[4/4] Generation des graphiques...")
    if metrics is not None and history_df is not None:
        plot_training_curves_and_confusion_matrix(history_df, metrics , model)
    else:
        print("Impossible de generer les graphiques:")
        if metrics is None:
            print("  - Metriques non disponibles")
        if history_df is None:
            print("  - Historique d entrainement non disponible")
    
    # ============================================
    # ETAPE 5: EXPORTATION ONNX (avec chemin auto)
    # ============================================
    print("\n[5/4] Exportation au format ONNX...")
    save_onnx_model(model, exp_path)
    
    # ============================================
    # RESUME FINAL
    # ============================================
    print("\n" + "=" * 60)
    print("ENTRAINEMENT TERMINE!")
    print("=" * 60)
    print(f"Modele ONNX: {OUTPUT_MODEL_PATH}")
    print(f"Graphiques:  {OUTPUT_CURVES_PATH}")
    print("=" * 60)


# ============================================
# POINT D'ENTREE DU PROGRAMME
# ============================================

if __name__ == "__main__":
    main()