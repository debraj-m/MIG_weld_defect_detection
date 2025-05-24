import os

base_labels_dir = r'C:\Users\debra\Desktop\Dataset\labels'

# List of subfolders to process
subfolders = ['train', 'val']

for folder in subfolders:
    folder_path = os.path.join(base_labels_dir, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.yolo'):
            base = os.path.splitext(filename)[0]
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, base + '.txt')
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

