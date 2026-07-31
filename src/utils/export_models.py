"""
Export du modèle best.pt vers NCNN (optimisé pour Raspberry Pi 4)
Avec précision maximale
"""

from ultralytics import YOLO
import torch

MODEL_PATH = '/media/mohamedaziz-hadjayed/D/aziz_data/fatigue_detection/edge-ai-wearable-fatigue-detection/runs/detect/runs/face_detection_yolov8n/weights/best.pt'

print("📂 Chargement du modèle...")
model = YOLO(MODEL_PATH)

# Vérifier les classes
print(f"📋 Classes: {model.names}")
print(f"📊 Nombre de classes: {len(model.names)}")

# ============ EXPORT NCNN (FP32) ============
print("\n🚀 Export vers NCNN (FP32 - meilleure précision)...")
ncnn_path = model.export(
    format='ncnn',
    imgsz=640,  # Garder 640 pour la précision
    half=False  # FP32, pas FP16
)

print(f"✅ NCNN exporté: {ncnn_path}")

# ============ EXPORT ONNX (FP32) ============
print("\n🚀 Export vers ONNX (FP32)...")
onnx_path = model.export(
    format='onnx',
    imgsz=640,
    simplify=False,  # Désactiver simplify pour garder la précision
    opset=12
)

print(f"✅ ONNX exporté: {onnx_path}")

# ============ EXPORT TFLite (FP32) ============
print("\n🚀 Export vers TFLite (FP32)...")
tflite_path = model.export(
    format='tflite',
    imgsz=640,
    int8=False  # FP32 pour meilleure précision
)

print(f"✅ TFLite exporté: {tflite_path}")

print("\n" + "="*60)
print("📊 RÉSUMÉ DES EXPORTS")
print("="*60)
print(f"📁 NCNN (FP32):  {ncnn_path}")
print(f"📁 ONNX (FP32):  {onnx_path}")
print(f"📁 TFLite (FP32): {tflite_path}")
print("="*60)
print("\n💡 Pour Raspberry Pi 4, utilisez NCNN (meilleur compromis)")