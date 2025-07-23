import cv2
from ultralytics import YOLO


# === CONFIG ===
# Update these paths as needed for your test images and model weights
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.abspath(os.path.join(BASE_DIR, '../test_images/Experiment Photos/PXL_20250525_094459478.MP.jpg'))  # Example image
defect_model_path = os.path.abspath(os.path.join(BASE_DIR, '../weights/weldingPlate.pt'))  # Example model
CONF_THRESHOLD = 0.25 # 🔧 Tuned confidence threshold
IOU_THRESHOLD = 0.7   # 🔧 Tuned NMS IoU threshold

class_names = {
    0: "spatter",
}

# === Load model ===
defect_model = YOLO(defect_model_path)

# === Load image ===
image = cv2.imread(image_path)
if image is None:
    print(f"❌ Could not load image: {image_path}")
    exit()

# === Defect detection with tuned thresholds ===
defect_results = defect_model(image_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
boxes = defect_results[0].boxes

# === Draw all detected defects ===
for i, box in enumerate(boxes.xyxy.tolist()):
    conf = boxes.conf[i].item()
    cls = int(boxes.cls[i].item())
    class_name = class_names.get(cls, f"Class {cls}")
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{class_name} ({conf:.2f})", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# === Live Display ===
cv2.imshow("Defect Detection Only", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === Optional Save ===
cv2.imwrite("weld_test_result.jpg", image)
print("✅ Saved final output to weld_test_result.jpg")


