"""
Pipeline temps réel : Détection de visage (YOLOv8n Detect) 
→ Classification Fatigue/Non-Fatigue (YOLOv8n Classify)
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# ============================================
# CONFIGURATION
# ============================================

DETECTION_MODEL_PATH = '/media/mohamedaziz-hadjayed/D/aziz_data/fatigue_detection/edge-ai-wearable-fatigue-detection/runs/detect/runs/face_detection_yolov8n/weights/best.pt'
CLASSIFICATION_MODEL_PATH = '/media/mohamedaziz-hadjayed/D/aziz_data/fatigue_detection/edge-ai-wearable-fatigue-detection/runs/classify/runs/classify/runs/train/yolov8n-cls/exp/weights/best.pt'

# Paramètres
CONF_THRESHOLD = 0.5      # Seuil confiance détection visage
CLS_CONF_THRESHOLD = 0.6  # Seuil confiance classification
IMG_SIZE_CLS = 224        # Taille entrée modèle classification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Couleurs (BGR)
COLOR_FATIGUE = (0, 0, 255)      # Rouge
COLOR_NON_FATIGUE = (0, 255, 0)  # Vert
COLOR_FACE_BOX = (255, 255, 0)   # Cyan
COLOR_TEXT = (255, 255, 255)     # Blanc
COLOR_DETECTION = (0, 255, 255)  # Jaune pour les infos de détection

# Historique pour lissage (moyenne glissante)
HISTORY_SIZE = 5
fatigue_history = deque(maxlen=HISTORY_SIZE)

# ============================================
# CHARGEMENT DES MODÈLES
# ============================================

print("=" * 60)
print("CHARGEMENT DES MODÈLES")
print("=" * 60)

print(f"\n[1/2] Chargement modèle détection (Device: {DEVICE})...")
det_model = YOLO(DETECTION_MODEL_PATH)
det_model.to(DEVICE)
print(f"✅ Détection chargée : {DETECTION_MODEL_PATH}")

# Récupérer les noms des classes du modèle de détection
det_class_names = det_model.names
print(f"📋 Classes de détection: {det_class_names}")

print(f"\n[2/2] Chargement modèle classification...")
cls_model = YOLO(CLASSIFICATION_MODEL_PATH)
cls_model.to(DEVICE)
print(f"✅ Classification chargée : {CLASSIFICATION_MODEL_PATH}")

# Récupérer les noms des classes du modèle de classification
cls_class_names = cls_model.names
print(f"📋 Classes de classification: {cls_class_names}")

print(f"\n{'=' * 60}")
print("MODÈLES PRÊTS - DÉMARRAGE WEBCAM")
print(f"{'=' * 60}")

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def preprocess_face_for_classification(face_roi, target_size=224):
    """
    Prétraite le ROI visage pour le modèle de classification.
    Redimensionne en 224x224 et normalise.
    """
    if face_roi is None or face_roi.size == 0:
        return None
    
    # Redimensionnement
    face_resized = cv2.resize(face_roi, (target_size, target_size))
    
    # Conversion BGR → RGB (YOLO attend RGB)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    return face_rgb


def classify_fatigue(face_roi):
    """
    Classifie un visage en Fatigue / Non-Fatigue.
    
    Returns:
        tuple: (label_str, confidence, is_fatigue_bool)
    """
    face_processed = preprocess_face_for_classification(face_roi, IMG_SIZE_CLS)
    
    if face_processed is None:
        return "Erreur", 0.0, False
    
    # Inférence classification
    results = cls_model(face_processed, verbose=False)
    
    if len(results) > 0 and results[0].probs is not None:
        probs = results[0].probs.data.cpu().numpy()
        
        # Mapping: YOLO classe 0 = fatigue, 1 = non_fatigue (ordre alphabétique)
        yolo_pred = np.argmax(probs)
        confidence = float(np.max(probs))
        
        if yolo_pred == 0:  # YOLO classe 0 = "fatigue" (alphabétique)
            label = "FATIGUE"
            is_fatigue = True
        else:
            label = "NON-FATIGUE"
            is_fatigue = False
        
        return label, confidence, is_fatigue
    
    return "Inconnu", 0.0, False


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=5):
    """Dessine un rectangle aux coins arrondis."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Lignes
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Coins
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_fatigue_bar(img, x, y, w, h, confidence, is_fatigue):
    """Dessine une barre de confiance style HUD."""
    bar_w = w
    bar_h = 8
    
    # Fond
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
    
    # Remplissage
    fill_w = int(bar_w * confidence)
    color = COLOR_FATIGUE if is_fatigue else COLOR_NON_FATIGUE
    cv2.rectangle(img, (x, y), (x + fill_w, y + bar_h), color, -1)
    
    # Bordure
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (200, 200, 200), 1)


# ============================================
# PIPELINE PRINCIPAL
# ============================================

def main():
    # Ouverture webcam (0 = caméra par défaut)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erreur: Impossible d'ouvrir la webcam")
        return
    
    # Configuration webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Résolution effective
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n📷 Webcam: {frame_w}x{frame_h}")
    
    # Variables de performance
    fps_history = deque(maxlen=30)
    prev_time = time.time()
    
    print("\n" + "=" * 60)
    print("CONTROLES:")
    print("  [Q] ou [ESC] → Quitter")
    print("  [S]          → Sauvegarder capture")
    print("=" * 60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erreur lecture frame")
            break
        
        # Copie pour affichage
        display = frame.copy()
        
        # ================================
        # ÉTAPE 1: DÉTECTION DES VISAGES
        # ================================
        det_results = det_model(frame, verbose=False, conf=CONF_THRESHOLD)
        
        faces_detected = 0
        fatigue_count = 0
        
        # Pour chaque visage détecté
        for result in det_results:
            boxes = result.boxes
            
            for box in boxes:
                # Coordonnées bbox
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_det = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Récupérer le nom de la classe détectée
                class_name = det_class_names.get(cls_id, f"Classe {cls_id}")
                
                # Vérification dimensions
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Extraction ROI visage
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.size == 0:
                    continue
                
                faces_detected += 1
                
                # ================================
                # ÉTAPE 2: CLASSIFICATION FATIGUE
                # ================================
                label, conf_cls, is_fatigue = classify_fatigue(face_roi)
                
                # Lissage par historique
                fatigue_history.append(1 if is_fatigue else 0)
                smoothed_fatigue = sum(fatigue_history) / len(fatigue_history) > 0.5
                
                if smoothed_fatigue:
                    fatigue_count += 1
                
                # Couleur selon résultat
                box_color = COLOR_FATIGUE if smoothed_fatigue else COLOR_NON_FATIGUE
                
                # ================================
                # AFFICHAGE
                # ================================
                
                # Rectangle visage (épaisseur dynamique selon confiance)
                thickness = 2 if conf_det > 0.8 else 1
                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, thickness)
                
                # Affichage du nom de la classe de détection
                det_label = f"{class_name}: {conf_det:.1%}"
                
                # Taille texte
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                thick = 2
                
                # Dimensions du texte de classification
                label_text = f"{label}"
                (tw, th), _ = cv2.getTextSize(label_text, font, scale, thick)
                
                # Fond pour le label de classification (en haut)
                pad = 6
                ly1 = y1 - th - pad * 2 - 25  # Espace pour barre + textes
                ly2 = y1
                
                # Assurer qu'on ne dépasse pas le haut de l'image
                if ly1 < 0:
                    ly1 = y2 + 5
                    ly2 = y2 + th + pad * 2 + 25
                
                # Dessin fond arrondi pour classification
                overlay = display.copy()
                cv2.rectangle(overlay, (x1, ly1), (x1 + max(tw + 80, 180), ly2), box_color, -1)
                cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
                
                # Texte label principal (FATIGUE / NON-FATIGUE)
                cv2.putText(display, label_text, 
                           (x1 + pad, ly2 - pad - 20), 
                           font, scale + 0.1, COLOR_TEXT, thick)
                
                # Confiance classification
                cv2.putText(display, f"Cls: {conf_cls:.1%}", 
                           (x1 + pad, ly2 - pad - 2), 
                           font, 0.45, (230, 230, 230), 1)
                
                # Barre de confiance
                bar_y = ly2 - 18
                draw_fatigue_bar(display, x1 + 80, bar_y, 90, 12, conf_cls, smoothed_fatigue)
                
                # Petit indicateur visuel (emoji style)
                emoji = "😴" if smoothed_fatigue else "😊"
                emoji_x = x1 + max(tw + 90, 170)
                cv2.putText(display, emoji, (emoji_x, ly2 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
                
                # ================================
                # AFFICHAGE DES INFOS DE DÉTECTION (en bas de la bbox)
                # ================================
                # Position pour les infos de détection (en bas du rectangle)
                det_y = y2 + 20
                
                # Assurer qu'on reste dans l'image
                if det_y + 20 > frame_h:
                    det_y = y1 - 30
                
                # Fond pour les infos de détection
                det_w, det_h = cv2.getTextSize(det_label, font, 0.45, 1)[0]
                cv2.rectangle(display, (x1, det_y - det_h - 4), 
                             (x1 + det_w + 8, det_y + 4), (0, 0, 0), -1)
                cv2.rectangle(display, (x1, det_y - det_h - 4), 
                             (x1 + det_w + 8, det_y + 4), (100, 100, 100), 1)
                
                # Texte de détection
                cv2.putText(display, det_label, (x1 + 4, det_y), 
                           font, 0.45, (0, 255, 255), 1)
                
                # ID de la classe en petit
                cls_id_text = f"ID:{cls_id}"
                cv2.putText(display, cls_id_text, (x1 + det_w + 12, det_y), 
                           font, 0.35, (150, 150, 150), 1)
        
        # ================================
        # HUD GLOBAL
        # ================================
        
        # Calcul FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)
        prev_time = current_time
        
        # Fond HUD
        hud_overlay = display.copy()
        cv2.rectangle(hud_overlay, (10, 10), (450, 130), (0, 0, 0), -1)
        cv2.addWeighted(hud_overlay, 0.6, display, 0.4, 0, display)
        
        # Infos HUD
        cv2.putText(display, "🔍 PIPELINE FATIGUE DETECTION", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        cv2.putText(display, f"FPS: {avg_fps:.1f} | Visages: {faces_detected}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        
        # Infos des modèles
        cv2.putText(display, f"Detect: {len(det_class_names)} classes | Classify: {len(cls_class_names)} classes", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Statut global
        if faces_detected > 0:
            if fatigue_count > 0:
                status = "⚠️ FATIGUE DETECTEE"
                status_color = COLOR_FATIGUE
            else:
                status = "✅ ALERTE"
                status_color = COLOR_NON_FATIGUE
        else:
            status = "👤 Aucun visage"
            status_color = (200, 200, 200)
        
        cv2.putText(display, status, (20, 108), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        
        # Barre FPS
        fps_ratio = min(avg_fps / 30.0, 1.0)
        fps_color = (0, 255, 0) if fps_ratio > 0.7 else (0, 165, 255) if fps_ratio > 0.4 else (0, 0, 255)
        cv2.rectangle(display, (20, 118), (20 + int(150 * fps_ratio), 126), fps_color, -1)
        cv2.rectangle(display, (20, 118), (170, 126), (200, 200, 200), 1)
        
        # ================================
        # AFFICHAGE
        # ================================
        cv2.imshow("Fatigue Detection - Pipeline YOLO", display)
        
        # Gestion touches
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q ou ESC
            break
        elif key == ord('s'):  # Screenshot
            filename = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, display)
            print(f"💾 Capture sauvegardée: {filename}")
    
    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Pipeline arrêté proprement")


# ============================================
# POINT D'ENTRÉE
# ============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interruption par l'utilisateur")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        cv2.destroyAllWindows()
        raise