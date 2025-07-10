import cv2
import numpy as np
import os
from feature_stats import GeometricFeatureStats

def extract_contour_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    enhanced = cv2.equalizeHist(denoised)
    smoothed = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(smoothed, 50, 150, apertureSize=3, L2gradient=True)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for i, contour in enumerate(contours):
        if len(contour) < 5 or cv2.contourArea(contour) < 50:
            continue
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        feature = {
            'contour_id': i,
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': float(max(w,h)) / min(w,h) if min(w,h) > 0 else 0,
            'solidity': float(area) / hull_area if hull_area > 0 else 0,
            'compactness': (perimeter**2) / (4*np.pi*area) if area > 0 else 0,
            'bbox': [x, y, x+w, y+h]
        }
        features.append(feature)
    return features

def detect_defects_by_features(image_path, stats_csv_path, defect_type='crack'):
    image = cv2.imread(image_path)
    stats = GeometricFeatureStats(stats_csv_path)
    features = extract_contour_features(image)
    detected = []
    for feat in features:
        if defect_type == 'porosity':
            in_range = (
                stats.is_within_range('area', feat['area']) and
                stats.is_within_range('aspect_ratio', feat['aspect_ratio']) and
                stats.is_within_range('solidity', feat['solidity']) and
                stats.is_within_range('compactness', feat['compactness'])
            )
        else:  # crack
            in_range = (
                stats.is_within_range('aspect_ratio', feat['aspect_ratio']) and
                stats.is_within_range('compactness', feat['compactness'])
            )
        if in_range:
            detected.append(feat)
    return detected

def visualize_and_save(image_path, detected, output_path):
    image = cv2.imread(image_path)
    for feat in detected:
        x1, y1, x2, y2 = [int(v) for v in feat['bbox']]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
        label = f"Defect"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    # Example usage
    image_path = r"Datasets/crack_dataset/train/images/15_jpeg_jpg.rf.ebdff550f0f589b1429574e4207503ea.jpg"
    stats_csv_path = os.path.join(os.path.dirname(__file__), 'crack_geometric_features.csv')
    detected = detect_defects_by_features(image_path, stats_csv_path, defect_type='crack')
    print(f"Detected {len(detected)} defects by geometric features.")
    visualize_and_save(image_path, detected, 'outputs/feature_based_detected.jpg')
