from ultralytics import YOLO
import cv2
import os
import csv

# Load models
seam_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weld_seam.pt")
defect_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weldDefect_drive_highacc.pt")

# Paths
input_dir = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\train\images"
raw_csv_path = "raw_detections.csv"
filtered_csv_path = "filtered_detections.csv"

# Class name mapping
class_names = {
    0: "Excess Reinforcement",
    1: "Crack",
    2: "Pore",
    3: "Spatter"
}

def is_within(defect_box, seam_box):
    dx1, dy1, dx2, dy2 = defect_box
    sx1, sy1, sx2, sy2 = seam_box
    return dx1 >= sx1 and dy1 >= sy1 and dx2 <= sx2 and dy2 <= sy2

def is_inside_any_seam(defect_box, seam_boxes):
    for seam_box in seam_boxes:
        if is_within(defect_box, seam_box):
            return True
    return False

# CSV headers
headers = ["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]

# Open both CSVs
with open(raw_csv_path, "w", newline="") as raw_file, \
     open(filtered_csv_path, "w", newline="") as filtered_file:

    raw_writer = csv.writer(raw_file)
    filtered_writer = csv.writer(filtered_file)
    raw_writer.writerow(headers)
    filtered_writer.writerow(headers)

    # Loop through images
    for img_file in os.listdir(input_dir):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(input_dir, img_file)

        # Raw YOLO defect detection (no filtering)
        defect_results = defect_model(img_path)
        defect_boxes = defect_results[0].boxes

        for i, box in enumerate(defect_boxes.xyxy.tolist()):
            conf = defect_boxes.conf[i].item()
            cls_id = int(defect_boxes.cls[i].item())
            class_name = class_names.get(cls_id, f"Class {cls_id}")
            raw_writer.writerow([img_file, cls_id, class_name, conf] + list(map(int, box)))

        # Geometry-aware filtering
        seam_results = seam_model(img_path)
        if not seam_results[0].boxes:
            continue
        seam_boxes = seam_results[0].boxes.xyxy.tolist()

        for i, box in enumerate(defect_boxes.xyxy.tolist()):
            if is_inside_any_seam(box, seam_boxes):
                conf = defect_boxes.conf[i].item()
                cls_id = int(defect_boxes.cls[i].item())
                class_name = class_names.get(cls_id, f"Class {cls_id}")
                filtered_writer.writerow([img_file, cls_id, class_name, conf] + list(map(int, box)))

print(f"\n✅ CSVs saved:\n- {raw_csv_path}\n- {filtered_csv_path}")
