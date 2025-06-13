import os

img_dir = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\Github_Dataset\train\images"
lbl_dir = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\Github_Dataset\train\labels"

for folder in [img_dir, lbl_dir]:
    for file in os.listdir(folder):
        if file.startswith("aug_"):
            os.remove(os.path.join(folder, file))
