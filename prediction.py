#To check an input image using a trained YOLOv8 model

from ultralytics import YOLO
import os
import cv2

# Load your trained model
model = YOLO(r'C:\Users\debra\Desktop\CODE\Dataset\Boundingbox.pt')  # Make sure best.pt is in the same directory or give full path

# Path to your input image
image_path = r"C:\Users\debra\Desktop\CODE\Dataset\Pictures from welding\PXL_20250525_094459478.MP.jpg"  # <-- Change to your image path
img=cv2.resize(cv2.imread(image_path),(640,640))
# Run prediction
results = model.predict(
    source=[img],
    conf=0.25,
    save=True,
    save_txt=True,
    imgsz=640,
    project=r"C:\Users\debra\Desktop\CODE\Dataset\results",  # <-- Your custom folder
    name="run1"  # Subfolder name inside 'my_results'
)

# Print prediction summary
for result in results:
    boxes = result.boxes
    print("\nDetected Objects:")
    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy[0].tolist()
        print(f"Class: {cls_id}, Confidence: {conf:.2f}, Box: {xyxy}")

print("\nAnnotated image saved to: runs/detect/predict/")