import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load the models
plate_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weldingPlate.pt")
defect_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\welddefect.pt")

def detect_and_process(image_path):
    # Load image
    img = cv2.imread(image_path)

    # Step 1: Detect weld plate
    plate_results = plate_model.predict(image_path, save=False, conf=0.5)

    if not plate_results or not plate_results[0].boxes:
        print("❌ No weld plate detected.")
        return

    boxes = plate_results[0].boxes.xyxy.cpu().numpy()
    classes = plate_results[0].boxes.cls.cpu().numpy()

    aligned = None
    for box, cls in zip(boxes, classes):
        if int(cls) == 0:  # Class 0 = weld plate
            x1, y1, x2, y2 = map(int, box)
            cropped = img[y1:y2, x1:x2]

            # Step 2: Rotate if vertical
            h, w = cropped.shape[:2]
            if h > w:
                aligned = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
                print("🔄 Rotated image for horizontal alignment.")
            else:
                aligned = cropped
            break

    if aligned is None:
        print("❌ No class 0 box found.")
        return

    # Step 3: Save cropped & aligned image
    os.makedirs("outputs", exist_ok=True)
    aligned_path = "outputs/aligned_plate.jpg"
    cv2.imwrite(aligned_path, aligned)
    print("✅ Cropped & aligned plate saved to", aligned_path)

    # Step 4: Run weld defect detection
    defect_results = defect_model.predict(
        aligned_path,
        save=True,
        conf=0.1,
        save_txt=True,
        project="outputs",
        name="labels"
    )

    print("✅ Defect detection output saved to outputs/labels")


if __name__ == "__main__":
    # 🔁 Provide your image path here
    image_path = r"C:\Users\debra\Desktop\CODE\Dataset\Pictures from welding\PXL_20250525_100532807.MP.jpg"
    detect_and_process(image_path)
