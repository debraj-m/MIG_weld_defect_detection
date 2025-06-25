from ultralytics import YOLO
import cv2
import os

# Paths
input_dir = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\train\images"
output_dir = "threshold_filtered"
os.makedirs(output_dir, exist_ok=True)

# Load models
seam_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weld_seam.pt")
defect_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weldDefect_drive_highacc.pt")

# Tuning parameters
CONF_THRESHOLD = 0.4   # default is 0.25
IOU_THRESHOLD = 0.5    # default is 0.7

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

# Inference loop
for img_file in os.listdir(input_dir):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(input_dir, img_file)
    image = cv2.imread(img_path)

    # Seam detection
    seam_result = seam_model(img_path)
    if not seam_result[0].boxes:
        print(f"⚠️ No seam detected in {img_file}")
        continue
    seam_boxes = seam_result[0].boxes.xyxy.tolist()

    # Defect detection with tuned thresholds
    defect_result = defect_model(img_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    boxes = defect_result[0].boxes
    filtered_boxes = []

    for i, box in enumerate(boxes.xyxy.tolist()):
        if is_inside_any_seam(box, seam_boxes):
            conf = boxes.conf[i].item()
            cls = int(boxes.cls[i].item())
            filtered_boxes.append((box, conf, cls))

    # Draw seam (blue) and filtered defects (green)
    for seam_box in seam_boxes:
        sx1, sy1, sx2, sy2 = map(int, seam_box)
        cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)

    for box, conf, cls in filtered_boxes:
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names.get(cls, cls)} ({conf:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_path = os.path.join(output_dir, img_file)
    cv2.imwrite(out_path, image)
    print(f"✅ Saved with thresholds to {out_path}")
