import cv2
from ultralytics import YOLO

# === CONFIG ===
image_path = r"C:\Users\debra\Desktop\IMG_3629_block_0_1_png_jpg.rf.08b179ce07867936aa212c4498fb6707.jpg"  # Replace with your test image
defect_model_path = r"C:\Users\debra\Downloads\best (16).pt"
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


