import os
import shutil

# Set paths
AUG_IMAGES_DIR = r"C:\Users\debra\Desktop\CODE\Dataset\Augmented\images"
AUG_LABELS_DIR = r"C:\Users\debra\Desktop\CODE\Dataset\Augmented\labels"

DEST_IMAGES_DIR = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\Github_Dataset\train\images"
DEST_LABELS_DIR = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\Github_Dataset\train\labels"

# Create destination folders if not exist
os.makedirs(DEST_IMAGES_DIR, exist_ok=True)
os.makedirs(DEST_LABELS_DIR, exist_ok=True)

# Move images
img_files = os.listdir(AUG_IMAGES_DIR)
for file in img_files:
    src = os.path.join(AUG_IMAGES_DIR, file)
    dst = os.path.join(DEST_IMAGES_DIR, file)
    shutil.move(src, dst)

# Move labels
label_files = os.listdir(AUG_LABELS_DIR)
for file in label_files:
    src = os.path.join(AUG_LABELS_DIR, file)
    dst = os.path.join(DEST_LABELS_DIR, file)
    shutil.move(src, dst)

print(f"✅ Transferred {len(img_files)} images and {len(label_files)} label files to training folders.")
