import cv2
import os

# Update this path to your dataset/images and labels
image_folder = r"C:\Users\debra\Desktop\Dataset\weld_quality_dataset\test\images"
label_folder = r"C:\Users\debra\Desktop\Dataset\weld_quality_dataset\test\labels"
output_folder = r"C:\Users\debra\Desktop\Dataset\weld_quality_dataset\annotated_images"  # e.g., "annotated_images"

# Class names from your YAML file
class_names = ['Crack', 'Excess Reinforcement', 'Porosity', 'Spatters', 'Welding seam']

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through images
for image_file in os.listdir(image_folder):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        image = cv2.imread(image_path)
        height, width, _ = image.shape

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    bbox_width = float(parts[3]) * width
                    bbox_height = float(parts[4]) * height

                    x1 = int(x_center - bbox_width / 2)
                    y1 = int(y_center - bbox_height / 2)
                    x2 = int(x_center + bbox_width / 2)
                    y2 = int(y_center + bbox_height / 2)

                    label = class_names[class_id]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

        # Save the annotated image
        cv2.imwrite(os.path.join(output_folder, image_file), image)

print("Done! Annotated images saved to:", output_folder)
