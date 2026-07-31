import os
import shutil
import random
from pathlib import Path

# Configuration
source_dir = "data/04_vision"
output_dir = "data/05_vision_splited/dataset"
train_ratio = 0.7
valid_ratio = 0.2
test_ratio = 0.1

# Créer les dossiers de destination
for split in ['train', 'valid', 'test']:
    for cls in ['fatigue', 'non_fatigue']:
        Path(f"{output_dir}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

# Traiter chaque classe
for class_name in ['fatigue', 'non_fatigue']:
    source_path = Path(source_dir) / class_name
    images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")) + list(source_path.glob("*.jpeg"))
    
    # Mélanger aléatoirement
    random.shuffle(images)
    
    # Calculer les indices de division
    total = len(images)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))
    
    # Assigner les images
    train_images = images[:train_end]
    valid_images = images[train_end:valid_end]
    test_images = images[valid_end:]
    
    print(f"\n{class_name}:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Valid: {len(valid_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    # Copier les images
    for img in train_images:
        shutil.copy2(img, f"{output_dir}/train/{class_name}/{img.name}")
    for img in valid_images:
        shutil.copy2(img, f"{output_dir}/valid/{class_name}/{img.name}")
    for img in test_images:
        shutil.copy2(img, f"{output_dir}/test/{class_name}/{img.name}")

print("\n✅ Dataset divisé avec succès !")
print(f"Structure créée dans : {output_dir}/")