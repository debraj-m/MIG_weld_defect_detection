import os
import cv2
from collections import defaultdict
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomBrightnessContrast,
    Rotate, Blur, CLAHE, Compose
)

# Paths
image_dir = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\Github_Dataset\train\images"
label_dir = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\Github_Dataset\train\labels"

# Desired count per class
target_counts = {
    "0": 17285,  # deposit
    "1": 17285,  # discontinuity
    "2": 17285   # pore
}

# Count current samples
current_counts = defaultdict(int)
file_map = defaultdict(list)

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file)) as f:
            lines = f.readlines()
        for line in lines:
            class_id = line.strip().split()[0]
            if class_id in target_counts:
                current_counts[class_id] += 1
                file_map[class_id].append(file)

# Augmentation
augmentations = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.3),
    RandomBrightnessContrast(p=0.4),
    Rotate(limit=15, p=0.4),
    Blur(p=0.2),
    CLAHE(p=0.3)
])

counter = defaultdict(int)
file_id = 0

for cid, files in file_map.items():
    needed = target_counts[cid] - current_counts[cid]
    print(f"\n📦 Class {cid} needs {needed} augmentations...")

    i = 0
    while counter[cid] < needed:
        base_file = files[i % len(files)]
        img_path = os.path.join(image_dir, base_file.replace(".txt", ".jpg"))
        lbl_path = os.path.join(label_dir, base_file)

        if not os.path.exists(img_path):
            i += 1
            continue

        image = cv2.imread(img_path)
        with open(lbl_path) as f:
            lines = f.readlines()
        # Keep only lines of this class
        lines = [line for line in lines if line.startswith(cid)]

        if not lines:
            i += 1
            continue

        aug = augmentations(image=image)
        aug_image = aug["image"]

        new_img_name = f"aug_{cid}_{file_id}.jpg"
        new_lbl_name = f"aug_{cid}_{file_id}.txt"

        cv2.imwrite(os.path.join(image_dir, new_img_name), aug_image)
        with open(os.path.join(label_dir, new_lbl_name), "w") as f_out:
            f_out.writelines(lines)

        counter[cid] += 1
        file_id += 1
        i += 1

print("\n✅ Final Balanced Augmentation Done:")
for cid in target_counts:
    print(f"Class {cid}: Augmented {counter[cid]} samples")
