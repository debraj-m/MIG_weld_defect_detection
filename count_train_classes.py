import os
from collections import defaultdict

label_dir = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\Github_Dataset\train\labels"  # 👈 only the training set
class_names = {
    "0": "deposit",
    "1": "discontinuity",
    "2": "pore",
    "3": "stain",
}

train_counts = defaultdict(int)

for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue
    with open(os.path.join(label_dir, file), "r") as f:
        for line in f:
            class_id = line.strip().split()[0]
            train_counts[class_id] += 1

print("📊 Class Count in Train Set Only:\n")
for class_id, count in sorted(train_counts.items()):
    print(f"{class_names.get(class_id, 'unknown')} (ID {class_id}): {count}")
