import cv2
import os
import csv
from ultralytics import YOLO
import numpy as np

# === CONFIG ===
plate_model_path = r"C:\Users\debra\Desktop\CODE\Dataset\weights\weldingPlate.pt"
seam_model_path = r"C:\Users\debra\Desktop\CODE\Dataset\weights\weld_seam.pt"
defect_model_path = r"C:\Users\debra\Desktop\CODE\Dataset\weights\weldDefect_drive_highacc.pt"
plate_class_id = 0
input_path = r"C:\Users\debra\Downloads\weld-defect-work-carried-out-260nw-2134513389.jpg"# Can be a folder OR a single .jpg/.png
output_dir = "results"
csv_path = "final_detections.csv"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

class_names = {
    0: "Excess Reinforcement",
    1: "Crack",
    2: "Pore",
    3: "Spatter"
}

os.makedirs(output_dir, exist_ok=True)

# === Load Models ===
plate_model = YOLO(plate_model_path)
seam_model = YOLO(seam_model_path)
defect_model = YOLO(defect_model_path)

def rotate_to_horizontal(image, box):
    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is not None:
        angle = np.rad2deg(np.median([line[0][1] for line in lines])) - 90
        (h, w) = cropped.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(cropped, M, (w, h), flags=cv2.INTER_LINEAR)
        return rotated
    return cropped

def is_within(defect_box, seam_box):
    dx1, dy1, dx2, dy2 = defect_box
    sx1, sy1, sx2, sy2 = seam_box
    return dx1 >= sx1 and dy1 >= sy1 and dx2 <= sx2 and dy2 <= sy2

def is_inside_any_seam(defect_box, seam_boxes):
    for seam_box in seam_boxes:
        if is_within(defect_box, seam_box):
            return True
    return False

# === Get Image List ===
if os.path.isdir(input_path):
    image_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
else:
    image_files = [input_path]

# === Output CSV ===
with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        orig = cv2.imread(img_path)
        if orig is None:
            print(f"❌ Could not load {img_path}")
            continue

        # Step 1: Weld Plate Detection
        plate_result = plate_model(img_path)
        plate_boxes = plate_result[0].boxes.xyxy.tolist()
        if not plate_boxes:
            print(f"⚠️ No plate found in {img_name} — continuing with full image")
            plate_crop = orig.copy()  # use full image
        else:
            plate_crop = rotate_to_horizontal(orig, plate_boxes[0])


        # Step 2: Seam Detection
        seam_result = seam_model(plate_crop)
        seam_boxes = seam_result[0].boxes.xyxy.tolist()
        if not seam_boxes:
            print(f"⚠️ No seam found in {img_name}")
            continue

        # Step 3: Defect Detection
        defect_result = defect_model(plate_crop, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
        boxes = defect_result[0].boxes

        for i, box in enumerate(boxes.xyxy.tolist()):
            if is_inside_any_seam(box, seam_boxes):
                conf = boxes.conf[i].item()
                cls = int(boxes.cls[i].item())
                class_name = class_names.get(cls, f"Class {cls}")
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(plate_crop, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(plate_crop, f"{class_name} ({conf:.2f})", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                writer.writerow([img_name, cls, class_name, conf, x1, y1, x2, y2])

        # Draw seam box
        for seam_box in seam_boxes:
            sx1, sy1, sx2, sy2 = map(int, seam_box)
            cv2.rectangle(plate_crop, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)

        # Save image
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, plate_crop)
        print(f"✅ Processed {img_name}")
