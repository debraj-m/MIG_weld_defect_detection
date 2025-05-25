#Splites the image into train and validation sets, moves them to respective folders
import os
import random
import shutil

# Set base paths
base_dir = os.getcwd()
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

# Output folders
img_train_dir = os.path.join(images_dir, 'train')
img_val_dir = os.path.join(images_dir, 'val')
lbl_train_dir = os.path.join(labels_dir, 'train')
lbl_val_dir = os.path.join(labels_dir, 'val')

# Create directories if not exist
for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
    os.makedirs(d, exist_ok=True)

# Get list of all images
all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle and split
random.seed(42)
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

def move_files(img_list, img_dst, lbl_dst):
    for img_file in img_list:
        img_path = os.path.join(images_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + '.yolo' 
        label_path = os.path.join(labels_dir, label_file)

        # Move image
        shutil.move(img_path, os.path.join(img_dst, img_file))

        # Move label if it exists
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(lbl_dst, label_file))
        else:
            print(f"⚠️ No label file found for {img_file}")

# Move train and val
move_files(train_images, img_train_dir, lbl_train_dir)
move_files(val_images, img_val_dir, lbl_val_dir)

# Create weld_defect.yaml
yaml_path = os.path.join(base_dir, 'weld_defect.yaml')
with open(yaml_path, 'w') as f:
    f.write(f"""train: {img_train_dir.replace(os.sep, '/')}
val: {img_val_dir.replace(os.sep, '/')}
nc: 4
names: ['deposit', 'stain', 'pore', 'discontinuity']
""")

print("✅ Split completed and `weld_defect.yaml` created.")
