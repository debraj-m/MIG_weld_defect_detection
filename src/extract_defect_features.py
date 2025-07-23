import os
import cv2
import numpy as np
import csv
from tqdm import tqdm

# Use the same feature extraction as in your main code
from prediction_final import analyze_contour_features

def extract_features_from_yolo_labels(image_dir, label_dir, class_name, output_rows):
    total_files = 0
    total_rois = 0
    for fname in tqdm(os.listdir(label_dir)):
        if not fname.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, fname)
        image_path = os.path.join(image_dir, fname.replace('.txt', '.jpg'))
        if not os.path.exists(image_path):
            image_path = os.path.join(image_dir, fname.replace('.txt', '.png'))
        if not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        rois_in_file = 0
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # YOLO format: class x_center y_center width height (normalized)
                _, xc, yc, bw, bh = map(float, parts[:5])
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                roi = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                # Skip empty or zero-size ROIs
                if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue
                features = analyze_contour_features(roi)
                for fdict in features:
                    row = [class_name, fname, x1, y1, x2, y2]
                    row += [
                        fdict['area'], fdict['aspect_ratio'], fdict['solidity'], fdict['compactness'],
                        fdict['circularity'], fdict['fill_ratio'], fdict['defects_count'],
                        fdict['mean_intensity'], fdict['std_intensity'], fdict['lbp_uniformity']
                    ]
                    output_rows.append(row)
                    rois_in_file += 1
        print(f"{class_name} - {fname}: {rois_in_file} ROIs found")
        total_files += 1
        total_rois += rois_in_file
    print(f"{class_name}: Processed {total_files} label files, found {total_rois} ROIs in total.")

def main():
    # Set your dataset paths
    base = r'c:/Users/debra/Desktop/CODE/Dataset/Datasets'
    datasets = [
        ('blowhole_dataset', 'blowhole'),
        ('porosity_dataset', 'porosity'),
    ]
    output_rows = []
    blowhole_rows = []
    porosity_rows = []
    n_blowhole = 0
    n_porosity = 0

    # First, extract all blowhole features
    for ds_folder, class_name in datasets:
        img_dir = os.path.join(base, ds_folder, 'train', 'images')
        lbl_dir = os.path.join(base, ds_folder, 'train', 'labels')
        temp_rows = []
        if class_name == 'blowhole':
            extract_features_from_yolo_labels(img_dir, lbl_dir, class_name, temp_rows)
            blowhole_rows.extend(temp_rows)
            n_blowhole = len(temp_rows)

    # Then, extract porosity features, but stop when enough are collected
    for ds_folder, class_name in datasets:
        if class_name != 'porosity':
            continue
        img_dir = os.path.join(base, ds_folder, 'train', 'images')
        lbl_dir = os.path.join(base, ds_folder, 'train', 'labels')
        porosity_count = 0
        for fname in tqdm(os.listdir(lbl_dir)):
            if porosity_count >= n_blowhole:
                break
            if not fname.endswith('.txt'):
                continue
            label_path = os.path.join(lbl_dir, fname)
            image_path = os.path.join(img_dir, fname.replace('.txt', '.jpg'))
            if not os.path.exists(image_path):
                image_path = os.path.join(img_dir, fname.replace('.txt', '.png'))
            if not os.path.exists(image_path):
                continue
            image = cv2.imread(image_path)
            if image is None:
                continue
            h, w = image.shape[:2]
            with open(label_path, 'r') as f:
                for line in f:
                    if porosity_count >= n_blowhole:
                        break
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    _, xc, yc, bw, bh = map(float, parts[:5])
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    roi = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                    if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                        continue
                    features = analyze_contour_features(roi)
                    for fdict in features:
                        if porosity_count >= n_blowhole:
                            break
                        row = [class_name, fname, x1, y1, x2, y2]
                        row += [
                            fdict['area'], fdict['aspect_ratio'], fdict['solidity'], fdict['compactness'],
                            fdict['circularity'], fdict['fill_ratio'], fdict['defects_count'],
                            fdict['mean_intensity'], fdict['std_intensity'], fdict['lbp_uniformity']
                        ]
                        porosity_rows.append(row)
                        porosity_count += 1
        n_porosity = porosity_count
        print(f"Porosity: Collected {n_porosity} ROIs to match blowhole count.")

    # Combine and shuffle
    output_rows = blowhole_rows + porosity_rows
    import random
    random.seed(42)
    random.shuffle(output_rows)

    # Write to CSV
    header = [
        'class','file','x1','y1','x2','y2','area','aspect_ratio','solidity','compactness',
        'circularity','fill_ratio','defects_count','mean_intensity','std_intensity','lbp_uniformity'
    ]
    import os
    output_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '../defect_features_balanced.csv'))
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(output_rows)
    print(f'Balanced feature extraction complete. Results saved to {output_csv}')

if __name__ == '__main__':
    main()
