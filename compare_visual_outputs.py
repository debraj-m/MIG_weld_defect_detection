import os
import cv2

# Directories
default_dir = "filtered_results"
tuned_dir = "threshold_filtered"
out_dir = "comparison_side_by_side"
os.makedirs(out_dir, exist_ok=True)

for img_file in os.listdir(default_dir):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    default_path = os.path.join(default_dir, img_file)
    tuned_path = os.path.join(tuned_dir, img_file)

    if not os.path.exists(tuned_path):
        print(f"❌ Skipping {img_file} (not found in both folders)")
        continue

    img1 = cv2.imread(default_path)
    img2 = cv2.imread(tuned_path)

    # Resize if not same shape
    if img1.shape != img2.shape:
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))

    combined = cv2.hconcat([img1, img2])

    # Add label bar
    label_bar = 50
    labeled = cv2.copyMakeBorder(combined, label_bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(labeled, "Default Thresholds", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(labeled, "Custom Thresholds", (img1.shape[1] + 30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    out_path = os.path.join(out_dir, img_file)
    cv2.imwrite(out_path, labeled)
    print(f"✅ Saved comparison: {out_path}")
