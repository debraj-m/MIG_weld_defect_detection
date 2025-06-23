from ultralytics import YOLO
import cv2
import os

# Load models
seam_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weld_seam.pt")
defect_model = YOLO(r"C:\Users\debra\Desktop\CODE\Dataset\weights\weldDefect_drive_highacc.pt")

# Input/output folders
input_dir = r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\train\images"
output_dir = "filtered_results"
os.makedirs(output_dir, exist_ok=True)

# Class name mapping (based on training order)
class_names = {
    0: "Excess Reinforcement",
    1: "Crack",
    2: "Pore",
    3: "Spatter"
}

# Check if defect bbox is fully inside seam bbox
def is_within(defect_box, seam_box):
    dx1, dy1, dx2, dy2 = defect_box
    sx1, sy1, sx2, sy2 = seam_box
    return dx1 >= sx1 and dy1 >= sy1 and dx2 <= sx2 and dy2 <= sy2

# Check if defect is inside any seam box
def is_inside_any_seam(defect_box, seam_boxes):
    for seam_box in seam_boxes:
        if is_within(defect_box, seam_box):
            return True
    return False

for img_file in os.listdir(input_dir):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(input_dir, img_file)
    image = cv2.imread(img_path)

    # Step 1: Detect seams
    seam_results = seam_model(img_path)
    if not seam_results[0].boxes:
        print(f"⚠️ No seam detected in {img_file}")
        continue
    seam_boxes = seam_results[0].boxes.xyxy.tolist()

    # Step 2: Detect defects
    defect_results = defect_model(img_path)
    boxes = defect_results[0].boxes
    filtered_boxes = []

    for i, box in enumerate(boxes.xyxy.tolist()):
        if is_inside_any_seam(box, seam_boxes):
            filtered_boxes.append((box, boxes.conf[i].item(), int(boxes.cls[i].item())))

    # Step 3: Draw seam boxes (in blue)
    for seam_box in seam_boxes:
        sx1, sy1, sx2, sy2 = map(int, seam_box)
        cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)  # Blue

    # Step 4: Draw filtered defect boxes (in green)
    for box, conf, cls in filtered_boxes:
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names.get(cls, f"Class {cls}")
        label = f"{class_name} ({conf:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Save result
    out_path = os.path.join(output_dir, img_file)
    cv2.imwrite(out_path, image)
    print(f"✅ Filtered image saved to {out_path}")
