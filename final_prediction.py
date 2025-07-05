import cv2
from ultralytics import YOLO

# === CONFIG ===
image_path = r"C:\Users\debra\Downloads\welding-porosity (2).jpg"# Replace with your test image
seam_model_path = r"C:\Users\debra\Desktop\CODE\Dataset\weights\weld_seam.pt"
defect_model_path = r"C:\Users\debra\Desktop\CODE\Dataset\weights\pore_detect.pt"
CONF_THRESHOLD = 0.4  # 🔧 Tuned confidence threshold
IOU_THRESHOLD = 0.7       # 🔧 Tuned NMS IoU threshold

class_names = {
    0: "pore",
}

# === Helper Functions ===
def is_within(defect_box, seam_box):
    dx1, dy1, dx2, dy2 = defect_box
    sx1, sy1, sx2, sy2 = seam_box
    return dx1 >= sx1 and dy1 >= sy1 and dx2 <= sx2 and dy2 <= sy2

def is_inside_any_seam(defect_box, seam_boxes):
    return any(is_within(defect_box, seam_box) for seam_box in seam_boxes)

# === Load models ===
seam_model = YOLO(seam_model_path)
defect_model = YOLO(defect_model_path)

# === Load image ===
image = cv2.imread(image_path)
if image is None:
    print(f"❌ Could not load image: {image_path}")
    exit()

# === Step 1: Seam detection ===
seam_results = seam_model(image_path)
seam_boxes = seam_results[0].boxes.xyxy.tolist()

# === Step 2: Defect detection with tuned thresholds ===
defect_results = defect_model(image_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
boxes = defect_results[0].boxes

# === Step 3: Geometry-aware filtering and drawing ===
if seam_boxes:
    # Only show defects inside seams
    for i, box in enumerate(boxes.xyxy.tolist()):
        if is_inside_any_seam(box, seam_boxes):
            conf = boxes.conf[i].item()
            cls = int(boxes.cls[i].item())
            class_name = class_names.get(cls, f"Class {cls}")
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{class_name} ({conf:.2f})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
else:
    print("⚠️ No seam detected. Showing all detected defects.")
    for i, box in enumerate(boxes.xyxy.tolist()):
        conf = boxes.conf[i].item()
        cls = int(boxes.cls[i].item())
        class_name = class_names.get(cls, f"Class {cls}")
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} ({conf:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# === Step 4: Draw seam box in blue ===
for seam_box in seam_boxes:
    sx1, sy1, sx2, sy2 = map(int, seam_box)
    cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)

# === Live Display ===
cv2.imshow("Final Detection (Tuned + Geometry-Aware)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === Optional Save ===
cv2.imwrite("weld_test_result.jpg", image)
print("✅ Saved final output to weld_test_result.jpg")


