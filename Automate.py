import os
import shutil
import random
from pathlib import Path

# === Config ===
SOURCE_DIR = r"C:\Users\debra\Downloads\weld-dataset (1)\weld-dataset\all welds"      # e.g., r"C:\Users\debra\Desktop\dataset\1"
DEST_DIR = r"C:\Users\debra\Downloads\weld-dataset (1)\weld-dataset" # e.g., r"C:\Users\debra\Desktop\dataset\processed"
SPLIT_RATIOS = (0.7, 0.2, 0.1)       # train, val, test

# === Setup Output Folders ===
for split in ['train', 'val', 'test']:
    os.makedirs(f"{DEST_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{DEST_DIR}/labels/{split}", exist_ok=True)

# === Gather All Images ===
all_images = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.jpg')]
random.shuffle(all_images)

# === Split ===
n = len(all_images)
train_split = int(n * SPLIT_RATIOS[0])
val_split = int(n * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))

splits = {
    'train': all_images[:train_split],
    'val': all_images[train_split:val_split],
    'test': all_images[val_split:]
}

# === Process Images and Labels ===
for split_name, files in splits.items():
    for img_file in files:
        base_name = os.path.splitext(img_file)[0]
        yolo_file = os.path.join(SOURCE_DIR, base_name + ".yolo")
        txt_file = os.path.join(SOURCE_DIR, base_name + ".txt")

        # Convert .yolo to .txt
        if os.path.exists(yolo_file):
            with open(yolo_file, 'r') as yf:
                data = yf.read()
            with open(txt_file, 'w') as tf:
                tf.write(data)

        # Copy image and label
        shutil.copy(os.path.join(SOURCE_DIR, img_file), f"{DEST_DIR}/images/{split_name}/{img_file}")
        if os.path.exists(txt_file):
            shutil.copy(txt_file, f"{DEST_DIR}/labels/{split_name}/{base_name}.txt")
