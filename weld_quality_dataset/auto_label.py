from ultralytics import YOLO
import os

# Load your trained model
model = YOLO(r"C:\Users\debra\Desktop\Dataset\weld_quality_dataset\best.pt")

# Path to folder containing images
image_dir = r"C:\Users\debra\Desktop\Dataset\weld_quality_dataset\train\images"

# Output directory for YOLO-format .txt labels
output_dir = os.path.join("C:/Users/debra/Desktop/Dataset/weld_quality_dataset/labels", "train")
os.makedirs(output_dir, exist_ok=True)

# Run prediction and save .txt files
results = model.predict(
    source=image_dir,
    save=False,
    save_txt=True,
    project="C:/Users/debra/Desktop/Dataset/weld_quality_dataset/labels",
    name="train",
    exist_ok=True
)

print("✅ Annotation complete! Check the 'labels/train' folder.")
