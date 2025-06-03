#To check an input image using a trained YOLOv8 model

from ultralytics import YOLO
import os

# Load your trained model
model = YOLO(r"C:\Users\debra\Desktop\Dataset\New_drivelink.pt")  # Make sure best.pt is in the same directory or give full path

# Path to your input image
image_path = r"C:\Users\debra\Downloads\PXL_20250525_094459478.MP.jpg"  # <-- Change to your image path

# Run prediction
results = model.predict(
    source=image_path,
    conf=0.01,        # Lower threshold to ensure detections appear
    save=True,       # Saves image with bounding boxes to runs/detect/predict
    save_txt=True,   # Optional: saves bounding box info as .txt
    imgsz=640        # Optional: match image size to training
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