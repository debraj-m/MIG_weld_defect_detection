from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image

# Load Models
plate_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weldingPlate.pt")     # Plate detection model
defect_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\welddefect.pt")      # Weld defect model

# Create output directory
os.makedirs("outputs", exist_ok=True)

def rotate_to_horizontal(image, box):
    """
    Rotates the cropped plate to be horizontally aligned
    based on its bounding box.
    """
    x1, y1, x2, y2 = map(int, box)

    crop = image[y1:y2, x1:x2]

    # Convert to grayscale and use Canny to find edges
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines with Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    angle = 0
    if lines is not None:
        for rho, theta in lines[0]:
            angle = np.rad2deg(theta) - 90
            break  # use the first line only

    # Rotate image to fix alignment
    (h, w) = crop.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR)

    return aligned

def detect_and_process(image_path):
    # Load image
    img = cv2.imread(image_path)

    # --- Stage 1: Detect Weld Plate ---
    results = plate_model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print("❌ No weld plate detected.")
        return

    # Assume largest box is the weld plate
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    largest_idx = np.argmax(areas)
    plate_box = boxes[largest_idx]

    # Rotate and crop plate
    cropped_plate = rotate_to_horizontal(img, plate_box)

    # Save for visual confirmation
    plate_path = "outputs/aligned_plate.jpg"
    cv2.imwrite(plate_path, cropped_plate)
    print(f"✅ Cropped & aligned plate saved to {plate_path}")

    # --- Stage 2: Detect Defects ---
    defect_results = defect_model(cropped_plate)

    # Save annotated result
    defect_results[0].save(filename="outputs/defect_detection.jpg")
    print("✅ Defect detection output saved to outputs/defect_detection.jpg")

    # Save YOLO format predictions
    defect_results[0].save_txt("outputs/defect_predictions.txt", save_conf=True)

    print("✅ YOLO predictions saved to outputs/labels/")

# Run on your image
detect_and_process(r"C:\Users\debra\Desktop\CODE\Dataset\Pictures from welding\PXL_20250525_094459478.MP.jpg")
