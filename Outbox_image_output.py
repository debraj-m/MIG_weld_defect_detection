from ultralytics import YOLO
import os

# Load the trained model
model = YOLO(r"C:\Users\debra\Desktop\Dataset\Result from drive link\weights\best.pt")
# Folder containing input images
image_folder = r"C:\Users\debra\Desktop\Dataset\weld_quality_dataset\train\images"
output_folder = r"C:\Users\debra\Desktop\Dataset\weld_quality_dataset\annotated_images_with_best.pt_file\train_images_output"
os.makedirs(output_folder, exist_ok=True)

# Run prediction on all images
results = model.predict(source=image_folder, save=True, save_txt=True, project=output_folder, name="predictions")

print("✅ Prediction complete. Annotated images and labels saved in:", os.path.join(output_folder, "predictions"))
