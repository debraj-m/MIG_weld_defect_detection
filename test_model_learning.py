import os
import cv2
from ultralytics import YOLO

# === Step 1: Load the model ===
model_path = r"C:\Users\debra\Downloads\best (9).pt"  # Update if needed
model = YOLO(model_path)
print(f"✅ Loaded model from: {model_path}")

# === Step 2: Inference on test image ===
test_image = r"C:\Users\debra\Desktop\CODE\Dataset\Test_images\Experiment Photos\PXL_20250525_104231099.jpg"  # Change this if needed
results = model(test_image, conf=0.1, save=True, project="test_output", name="test", verbose=True)

# === Step 3: Analyze and print detection boxes ===
boxes = results[0].boxes
if not boxes or len(boxes) == 0:
    print("❌ No detections found by the model.")
else:
    print(f"✅ {len(boxes)} detection(s) found:")
    for i, box in enumerate(boxes):
        xyxy = box.xyxy.tolist()
        conf = float(box.conf)
        cls = int(box.cls)
        print(f" 🔹 Detection {i+1}: Class={cls}, Confidence={conf:.3f}, Box={xyxy}")

# === Step 4: Save visual result ===
output_image_path = os.path.join("test_output", "test", os.path.basename(test_image))
print(f"📸 Output image with boxes saved to: {output_image_path}")

# === Step 5: Optional — show output image ===
img = cv2.imread(output_image_path)
if img is not None:
    cv2.imshow("YOLO Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("⚠️ Could not load the output image for display.")
