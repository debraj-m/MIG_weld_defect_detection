# This script detects weld plates, extracts weld seams, and identifies defects using YOLO models.
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load the models
plate_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weldingPlate.pt")
seam_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weld_seam.pt")  # NEW
defect_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\welddefect_github.pt")

def detect_and_process(image_path):
    # Load image
    img = cv2.imread(image_path)

    # Step 1: Detect weld plate
    plate_results = plate_model.predict(image_path, save=False, conf=0.5)
    if not plate_results or not plate_results[0].boxes:
        print("⚠️ No weld plate detected. Trying to detect weld seam on the original image.")
        # Try seam detection on the original image
        seam_results = seam_model.predict(image_path, save=False, conf=0.3)
        if not seam_results or not seam_results[0].boxes:
            print("❌ No weld seam detected on the original image. Running defect detection on the original image.")
            defect_results = defect_model.predict(
                image_path,
                save=True,
                conf=0.1,
                save_txt=True,
                project="outputs",
                name="labels"
            )
            print("✅ Defect detection output saved to outputs/labels")
            return

        seam_boxes = seam_results[0].boxes.xyxy.cpu().numpy()
        seam_classes = seam_results[0].boxes.cls.cpu().numpy()

        weld_seam = None
        for box, cls in zip(seam_boxes, seam_classes):
            if int(cls) == 0:  # Assuming class 0 = weld seam
                x1, y1, x2, y2 = map(int, box)
                weld_seam = img[y1:y2, x1:x2]
                break

        if weld_seam is None:
            print("❌ No class 0 weld seam box found on the original image.")
            return

        os.makedirs("outputs", exist_ok=True)
        weld_seam_path = "outputs/weld_seam.jpg"
        cv2.imwrite(weld_seam_path, weld_seam)
        print("✅ Weld seam cropped and saved to", weld_seam_path)

        # Step 3: Detect defects on cropped weld seam
        defect_results = defect_model.predict(
            weld_seam_path,
            save=True,
            conf=0.25,
            save_txt=True,
            project="outputs",
            name="labels"
        )
        print("✅ Defect detection output saved to outputs/labels")
        return

    boxes = plate_results[0].boxes.xyxy.cpu().numpy()
    classes = plate_results[0].boxes.cls.cpu().numpy()

    aligned = None
    for box, cls in zip(boxes, classes):
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            cropped = img[y1:y2, x1:x2]
            h, w = cropped.shape[:2]
            aligned = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE) if h > w else cropped
            break

    if aligned is None:
        print("❌ No weld plate found.")
        return

    os.makedirs("outputs", exist_ok=True)
    aligned_path = "outputs/aligned_plate.jpg"
    cv2.imwrite(aligned_path, aligned)
    print("✅ Cropped & aligned plate saved to", aligned_path)

    # Step 2: Detect weld seam from aligned plate
    seam_results = seam_model.predict(aligned_path, save=False, conf=0.3)
    if not seam_results or not seam_results[0].boxes:
        print("❌ No weld seam detected.")
        return

    seam_boxes = seam_results[0].boxes.xyxy.cpu().numpy()
    seam_classes = seam_results[0].boxes.cls.cpu().numpy()

    weld_seam = None
    for box, cls in zip(seam_boxes, seam_classes):
        if int(cls) == 0:  # Assuming class 0 = weld seam
            x1, y1, x2, y2 = map(int, box)
            weld_seam = aligned[y1:y2, x1:x2]
            break

    if weld_seam is None:
        print("❌ No class 0 weld seam box found.")
        return

    weld_seam_path = "outputs/weld_seam.jpg"
    cv2.imwrite(weld_seam_path, weld_seam)
    print("✅ Weld seam cropped and saved to", weld_seam_path)

    # Step 3: Detect defects on cropped weld seam
    defect_results = defect_model.predict(
        weld_seam_path,
        save=True,
        conf=0.5,
        save_txt=True,
        project="outputs",
        name="labels"
    )
    print("✅ Defect detection output saved to outputs/labels")


if __name__ == "__main__":
    image_path = r"C:\Users\debra\Downloads\welding-porosity.jpg"
    detect_and_process(image_path)
