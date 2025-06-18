import os
import cv2
import random
import shutil
from pathlib import Path
from collections import defaultdict
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomBrightnessContrast, Rotate, Compose
)

# 🔧 CONFIG
CLASS_TARGET = [0, 1, 2, 3]  # Class IDs to balance
TARGET_PER_CLASS = 18000     # Final count of each class's objects
LABEL_DIR = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\Github_Dataset\train\labels"
IMAGE_DIR = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\Github_Dataset\train\images"
AUG_SAVE_DIR = r"C:\Users\debra\Desktop\CODE\Dataset\Augmented"

# 📦 Setup output folders
os.makedirs(f"{AUG_SAVE_DIR}/images", exist_ok=True)
os.makedirs(f"{AUG_SAVE_DIR}/labels", exist_ok=True)

# 🧠 Augmentations
augment = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.3),
    Rotate(limit=25, p=0.5),
    RandomBrightnessContrast(p=0.4)
])

# 🧮 Count labels
class_counts = defaultdict(int)
class_files = defaultdict(list)

for label_file in os.listdir(LABEL_DIR):
    path = os.path.join(LABEL_DIR, label_file)
    with open(path, "r") as f:
        for line in f:
            cls = int(line.strip().split()[0])
            class_counts[cls] += 1
            if label_file not in class_files[cls]:
                class_files[cls].append(label_file)

print("📊 Initial class counts:", dict(class_counts))

# 🚀 Start augmenting
for cls in CLASS_TARGET:
    current_count = class_counts[cls]
    needed = TARGET_PER_CLASS - current_count
    print(f"\n🎯 Augmenting class {cls}: Need {needed} more samples")

    if needed <= 0:
        continue

    samples = class_files[cls]
    i = 0
    while needed > 0:
        label_file = random.choice(samples)
        image_file = label_file.replace('.txt', '.jpg')
        label_path = os.path.join(LABEL_DIR, label_file)
        image_path = os.path.join(IMAGE_DIR, image_file)

        # Skip if image not found
        if not os.path.exists(image_path):
            continue

        # Read and filter labels for only the current class
        with open(label_path, 'r') as f:
            lines = f.readlines()
        target_lines = [l for l in lines if int(l.strip().split()[0]) == cls]
        if not target_lines:
            continue

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            continue

        height, width = img.shape[:2]

        # Apply augmentation
        augmented = augment(image=img)
        aug_img = augmented["image"]

        # Save augmented image and filtered label
        new_image_name = f"{cls}_aug_{i}.jpg"
        new_label_name = f"{cls}_aug_{i}.txt"
        cv2.imwrite(os.path.join(AUG_SAVE_DIR, "images", new_image_name), aug_img)

        with open(os.path.join(AUG_SAVE_DIR, "labels", new_label_name), 'w') as f:
            for line in target_lines:
                f.write(line)

        i += 1
        needed -= len(target_lines)

    print(f"✅ Class {cls} augmented to {TARGET_PER_CLASS} objects.")

print("\n🎉 Augmentation done. You can now move `Augmented/` into your training folder.")
