"""
YOLOv8n Face Detection Training (4 classes)
RTX 4060 (8GB VRAM) optimized
"""

import os
# pyrefly: ignore [missing-import]
import torch
import yaml
from pathlib import Path
# pyrefly: ignore [missing-import]
from ultralytics import YOLO


# ============ CONFIG ============
DATASET_PATH = '/media/mohamedaziz-hadjayed/D/aziz_data/fatigue_detection/edge-ai-wearable-fatigue-detection/data/06_face_detection'  # Ton chemin Roboflow
YAML_FILE = f'{DATASET_PATH}/data.yaml'
EPOCHS = 150
BATCH_SIZE = 8  # RTX 4060 8GB → batch 8 est safe
IMGSZ = 640
DEVICE = 0  # GPU 0
OUTPUT_DIR = 'runs/face_detection'

# ============ VERIFY DATASET ============
print("📂 Vérification dataset...")
if not os.path.exists(YAML_FILE):
    print(f"❌ data.yaml non trouvé: {YAML_FILE}")
    exit(1)

with open(YAML_FILE) as f:
    data_config = yaml.safe_load(f)
    print(f"✅ Classes: {data_config['names']}")
    print(f"✅ nc: {data_config['nc']}")

# ============ LOAD MODEL ============
print("\n🔧 Chargement YOLOv8n...")
model = YOLO('yolov8n.pt')  # Nano model (léger, rapide)
print(f"✅ Modèle chargé")

# ============ TRAINING ============
print(f"\n🚀 Démarrage training...\n")

results = model.train(
    # Data
    data=YAML_FILE,
    
    # Hyperparams
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH_SIZE,
    
    # Optimization
    lr0=0.001,           # Learning rate initial
    lrf=0.01,            # LR final (10x reduction)
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    
    # Augmentation (YOLOv8 integrated)
    hsv_h=0.015,         # HSV hue
    hsv_s=0.7,           # HSV saturation
    hsv_v=0.4,           # HSV value
    degrees=10,          # Rotation
    translate=0.1,       # Translation
    scale=0.5,           # Zoom
    flipud=0.5,          # Flip vertical
    fliplr=0.5,          # Flip horizontal
    mosaic=1.0,          # Mosaic augmentation
    mixup=0.0,           # Mixup (disabled)
    
    # Early stopping & checkpointing
    patience=30,         # Early stop si val loss constant 30 epochs
    save_period=10,      # Save every 10 epochs
    
    # Device & optimization
    device=DEVICE,
    workers=4,           # DataLoader workers
    
    # Output
    project='runs',
    name='face_detection_yolov8n',
    save=True,
    exist_ok=False,      # Créer nouveau dossier
)

# ============ VALIDATION ============
print("\n📊 Évaluation sur validation set...")
metrics = model.val(data=YAML_FILE, imgsz=IMGSZ)

print(f"\n✅ mAP50: {metrics.box.map50:.4f}")
print(f"✅ mAP50-95: {metrics.box.map:.4f}")
print(f"✅ Precision: {metrics.box.mp:.4f}")
print(f"✅ Recall: {metrics.box.mr:.4f}")

# ============ TEST SET EVAL ============
print("\n📊 Évaluation sur test set...")
results_test = model.val(
    data=YAML_FILE,
    imgsz=IMGSZ,
    split='test'  # Si data.yaml inclut test
)

# ============ EXPORT ONNX (pour conversion TFLite après) ============
print("\n📦 Export ONNX...")
onnx_path = model.export(format='onnx', imgsz=IMGSZ, simplify=True)
print(f"✅ ONNX exporté: {onnx_path}")

# ============ EXPORT TFLITE INT8 (Raspberry Pi) ============
print("\n📦 Export TFLite INT8 (RPi optimization)...")
try:
    tflite_path = model.export(format='tflite', imgsz=416, int8=True)
    print(f"✅ TFLite INT8 exporté: {tflite_path}")
    print(f"   Resolution: 416x416 (optimisé RPi)")
    print(f"   Quantization: INT8 (plus rapide, moins précis)")
except Exception as e:
    print(f"⚠️  TFLite INT8 export failed: {e}")
    print("   Fallback: export sans INT8...")
    tflite_path = model.export(format='tflite', imgsz=416)
    print(f"✅ TFLite exporté (float): {tflite_path}")

# ============ SUMMARY ============
print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print("="*60)
print(f"\n📁 Best model: runs/face_detection_yolov8n/weights/best.pt")
print(f"📁 Last model: runs/face_detection_yolov8n/weights/last.pt")
print(f"📊 Results: runs/face_detection_yolov8n/results.csv")
print(f"\n🔌 Export locations:")
print(f"   - ONNX: {onnx_path}")
print(f"   - TFLite: {tflite_path}")
print(f"\n🚀 Next steps:")
print(f"   1. Vérifier mAP50 > 0.75 (bon)")
print(f"   2. Copier best.pt → RPi")
print(f"   3. Convertir best.pt → TFLite INT8")
print(f"   4. Intégrer dans FastAPI inference server")