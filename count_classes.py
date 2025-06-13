import os
from collections import defaultdict

# === CONFIG: Set the root of your dataset ===
dataset_root = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\Github_Dataset"  # e.g., "./weld_defect_dataset"
subfolders = ["train", "valid", "test"]

# Optional: Class ID to name mapping
class_names = {
    "0": "deposit",
    "1": "discontinuity",
    "2": "pore",
    "3": "stain",
}

# Initialize counter
class_counts = defaultdict(int)

# Traverse all subfolders
for split in subfolders:
    label_dir = os.path.join(dataset_root, split, "labels")
    if not os.path.exists(label_dir):
        continue

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, file), "r") as f:
            for line in f:
                class_id = line.strip().split()[0]
                class_counts[class_id] += 1

# Display final results
print("📊 Class Distribution Across Train/Valid/Test:\n")
for class_id, count in sorted(class_counts.items()):
    name = class_names.get(class_id, f"Class {class_id}")
    print(f"Class '{name}' (ID {class_id}): {count} objects")
