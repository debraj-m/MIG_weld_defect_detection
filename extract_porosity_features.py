import os
import cv2
import numpy as np
import pandas as pd

def yolo_to_bbox(yolo_line, img_w, img_h):
    # Only use the first 5 values, ignore extras
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Label line has fewer than 5 values: {yolo_line}")
    cls, x_c, y_c, w, h = map(float, parts[:5])
    x_c, y_c, w, h = x_c * img_w, y_c * img_h, w * img_w, h * img_h
    x1 = int(x_c - w / 2)
    y1 = int(y_c - h / 2)
    x2 = int(x_c + w / 2)
    y2 = int(y_c + h / 2)
    return int(cls), x1, y1, x2, y2

def extract_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area == 0:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0
    perimeter = cv2.arcLength(cnt, True)
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area != 0 else 0
    return dict(area=area, aspect_ratio=aspect_ratio, solidity=solidity, compactness=compactness)

def main():
    base_dir = os.path.join('Datasets', 'porosity_dataset')
    splits = ['train', 'val']
    output_csv = 'porosity_geometric_features.csv'
    results = []
    for split in splits:
        images_dir = os.path.join(base_dir, split, 'images')
        labels_dir = os.path.join(base_dir, split, 'labels')
        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            print(f"Skipping {split}: images or labels dir not found.")
            continue
        for fname in os.listdir(images_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                continue
            img_path = os.path.join(images_dir, fname)
            label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + '.txt')
            if not os.path.exists(label_path):
                print(f"No label for {img_path}")
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            img_h, img_w = img.shape
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f"Label file empty: {label_path}")
                for idx, line in enumerate(lines):
                    cls, x1, y1, x2, y2 = yolo_to_bbox(line, img_w, img_h)
                    crop = img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    if crop.size == 0:
                        print(f"Empty crop for {img_path}, bbox: {x1},{y1},{x2},{y2}")
                        continue
                    _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    features = extract_features(mask)
                    if not features:
                        print(f"No contours for {img_path}, bbox: {x1},{y1},{x2},{y2}")
                    if features:
                        features.update({
                            'split': split,
                            'image': fname,
                            'defect_idx': idx,
                            'class': int(cls),
                            'bbox_x1': x1,
                            'bbox_y1': y1,
                            'bbox_x2': x2,
                            'bbox_y2': y2
                        })
                        results.append(features)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved features for {len(df)} defects to {output_csv}")

if __name__ == '__main__':
    main()
