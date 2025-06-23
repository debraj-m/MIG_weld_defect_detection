import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load models
plate_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weldingPlate.pt")
defect_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weldDefect_drive_highacc.pt")

def crop_plate_and_detect_defects(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return

    os.makedirs("outputs", exist_ok=True)

    # Step 1: Detect weld plate
    plate_results = plate_model.predict(image_path, save=False, conf=0.5)
    plate_boxes = plate_results[0].boxes if plate_results and plate_results[0].boxes is not None else None

    if not plate_boxes or len(plate_boxes) == 0:
        print("❌ No weld plate detected.")
        return

    # Step 2: Crop & align the first detected plate (class 0)
    for box, cls in zip(plate_boxes.xyxy.cpu().numpy(), plate_boxes.cls.cpu().numpy()):
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            cropped = img[y1:y2, x1:x2]
            h, w = cropped.shape[:2]
            aligned = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE) if h > w else cropped
            aligned_path = "outputs/aligned_plate.jpg"
            cv2.imwrite(aligned_path, aligned)
            print("✅ Cropped & aligned weld plate saved to:", aligned_path)

            # Step 3: Defect Detection
            defect_model.predict(
                aligned_path,
                save=True,
                conf=0.05,
                save_txt=True,
                project="outputs",
                name="defect_results"
            )
            print("✅ Defect detection output saved to outputs/defect_results")
            return

    print("❌ No valid class 0 weld plate found.")

if __name__ == "__main__":
    image_path = r"C:\Users\debra\Desktop\CODE\Dataset\Test_images\Experiment Photos\PXL_20250525_113332348.jpg"
    crop_plate_and_detect_defects(image_path)
