import os

# Set your dataset folder path
dataset_path = r"C:\Users\debra\Desktop\CODE\Dataset\Drive_Dataset_Annotated\train"

# Check for 'images' and 'labels' folders
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

# Function to list files in a folder with given extensions
def list_files(folder, extensions):
    return [f for f in os.listdir(folder) if f.lower().endswith(extensions)]

# Check existence
if not os.path.exists(images_path):
    print("❌ 'images' folder is missing.")
else:
    image_files = list_files(images_path, ('.jpg', '.jpeg', '.png'))
    print(f"✅ Found {len(image_files)} image(s) in 'images' folder.")

if not os.path.exists(labels_path):
    print("❌ 'labels' folder is missing.")
else:
    label_files = list_files(labels_path, ('.txt',))
    print(f"✅ Found {len(label_files)} label file(s) in 'labels' folder.")
