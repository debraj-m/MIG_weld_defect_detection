import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load the models
plate_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weldingPlate.pt")
seam_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weld_seam.pt")
defect_model = YOLO(r"C:\Users\debra\Downloads\best (12).pt")

def detect_and_process(image_path):
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
        print("⚠️ No weld plate detected. Trying weld seam detection directly on original image.")

        # Try seam detection on the original image
        seam_results = seam_model.predict(image_path, save=False, conf=0.3)
        seam_boxes = seam_results[0].boxes if seam_results and seam_results[0].boxes is not None else None

        if not seam_boxes or len(seam_boxes) == 0:
            print("❌ No weld seam detected either. Running defect detection on original image.")
            defect_model.predict(
                image_path,
                save=True,
                conf=0.1,
                save_txt=True,
                project="outputs",
                name="labels"
            )
            print("✅ Defect detection output saved to outputs/labels")
            return

        # Crop first weld seam (class 0)
        for box, cls in zip(seam_results[0].boxes.xyxy.cpu().numpy(), seam_results[0].boxes.cls.cpu().numpy()):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                weld_seam = img[y1:y2, x1:x2]
                weld_seam_path = "outputs/weld_seam.jpg"
                cv2.imwrite(weld_seam_path, weld_seam)
                print("✅ Weld seam cropped and saved to", weld_seam_path)

                # Detect defects
                defect_model.predict(
                    weld_seam_path,
                    save=True,
                    conf=0.1,
                    save_txt=True,
                    project="outputs",
                    name="labels"
                )
                print("✅ Defect detection output saved to outputs/labels")
                return

        print("❌ No class 0 weld seam box found.")
        return

    # Crop and align weld plate
    for box, cls in zip(plate_boxes.xyxy.cpu().numpy(), plate_boxes.cls.cpu().numpy()):
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            cropped = img[y1:y2, x1:x2]
            h, w = cropped.shape[:2]
            aligned = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE) if h > w else cropped
            aligned_path = "outputs/aligned_plate.jpg"
            cv2.imwrite(aligned_path, aligned)
            print("✅ Cropped & aligned plate saved to", aligned_path)
            break
    else:
        print("❌ No valid weld plate found.")
        return

    # Step 2: Detect weld seam from aligned plate
    seam_results = seam_model.predict(aligned_path, save=False, conf=0.3)
    seam_boxes = seam_results[0].boxes if seam_results and seam_results[0].boxes is not None else None

    if not seam_boxes or len(seam_boxes) == 0:
        print("❌ No weld seam detected.")
        return

    # Crop first weld seam box (class 0)
    for box, cls in zip(seam_results[0].boxes.xyxy.cpu().numpy(), seam_results[0].boxes.cls.cpu().numpy()):
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            weld_seam = aligned[y1:y2, x1:x2]
            weld_seam_path = "outputs/weld_seam.jpg"
            cv2.imwrite(weld_seam_path, weld_seam)
            print("✅ Weld seam cropped and saved to", weld_seam_path)

            # Step 3: Defect Detection
            defect_model.predict(
                weld_seam_path,
                save=True,
                conf=0.1,
                save_txt=True,
                project="outputs",
                name="labels"
            )
            print("✅ Defect detection output saved to outputs/labels")
            return

    print("❌ No class 0 weld seam box found in aligned plate.")

if __name__ == "__main__":
    image_path = r"C:\Users\debra\Desktop\CODE\Dataset\Test_images\Experiment Photos\PXL_20250525_104231099.jpg"
    detect_and_process(image_path)
