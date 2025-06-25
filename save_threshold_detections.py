from ultralytics import YOLO
import os
import csv

# Load models
seam_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weld_seam.pt")
defect_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weldDefect_drive_highacc.pt")

# Input/output
input_dir = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\train\images"
csv_path = "threshold_detections.csv"

# Thresholds to test
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

# Class ID to Name mapping
class_names = {
    0: "Excess Reinforcement",
    1: "Crack",
    2: "Pore",
    3: "Spatter"
}

# Check if defect box is inside seam box
def is_within(defect_box, seam_box):
    dx1, dy1, dx2, dy2 = defect_box
    sx1, sy1, sx2, sy2 = seam_box
    return dx1 >= sx1 and dy1 >= sy1 and dx2 <= sx2 and dy2 <= sy2

def is_inside_any_seam(defect_box, seam_boxes):
    for seam_box in seam_boxes:
        if is_within(defect_box, seam_box):
            return True
    return False

# Prepare CSV
with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

    # Loop through images
    for img_file in os.listdir(input_dir):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(input_dir, img_file)

        # Step 1: detect seam
        seam_results = seam_model(img_path)
        if not seam_results[0].boxes:
            continue
        seam_boxes = seam_results[0].boxes.xyxy.tolist()

        # Step 2: detect defects with tuned thresholds
        defect_results = defect_model(img_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
        boxes = defect_results[0].boxes

        for i, box in enumerate(boxes.xyxy.tolist()):
            if is_inside_any_seam(box, seam_boxes):
                conf = boxes.conf[i].item()
                cls = int(boxes.cls[i].item())
                class_name = class_names.get(cls, f"Class {cls}")
                writer.writerow([img_file, cls, class_name, conf] + list(map(int, box)))

print(f"✅ Saved threshold-based detections to '{csv_path}'")
