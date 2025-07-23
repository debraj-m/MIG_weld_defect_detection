
from ultralytics import YOLO
import cv2
import sys
import joblib
import numpy as np
from sklearn.cluster import DBSCAN

# Import configuration
from .config import weights, confidence_thresholds, classifier_path, scaler_path, label_encoder_path




# Load trained classifier, scaler, and label encoder for porosity vs blowhole
try:
    clf = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
except Exception:
    clf = None
    scaler = None
    label_encoder = None

# Use weights from config
models = weights

def detect_with_model(model_path, image, threshold=0.5):
    model = YOLO(model_path)
    results = model(image)
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        # Filter by confidence threshold
        filtered_indices = confidences >= threshold
        return boxes[filtered_indices], confidences[filtered_indices]
    return [], []

def analyze_contour_features(roi):
    """Analyze geometric features of contours in a region of interest"""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    # Preprocessing
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    enhanced = cv2.equalizeHist(denoised)
    smoothed = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(smoothed, 50, 150, apertureSize=3, L2gradient=True)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 10]
    
    features = []
    roi_area = roi.shape[0] * roi.shape[1]
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        if area < 10:
            continue

        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        aspect_ratio = width / height if height > 0 else 1.0
        solidity = area / hull_area if hull_area > 0 else 0
        compactness = (perimeter * perimeter) / area if area > 0 else 0
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0

        # Bounding box fill ratio
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        fill_ratio = area / bbox_area if bbox_area > 0 else 0

        # Normalized features
        norm_area = area / roi_area if roi_area > 0 else 0
        norm_perimeter = perimeter / (2 * (roi.shape[0] + roi.shape[1])) if (roi.shape[0] + roi.shape[1]) > 0 else 0

        # Convexity defects (robust to OpenCV errors)
        defects_count = 0
        if len(contour) >= 4:
            try:
                hull_indices = cv2.convexHull(contour, returnPoints=False)
                if hull_indices is not None and len(hull_indices) > 3:
                    defects = cv2.convexityDefects(contour, hull_indices)
                    if defects is not None:
                        defects_count = defects.shape[0]
            except Exception:
                defects_count = 0

        # Intensity stats (mean, std) inside contour mask
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        std_intensity = np.std(gray[mask == 255])

        # Texture feature: Local Binary Pattern (LBP)
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
            lbp_masked = lbp[mask == 255]
            lbp_hist, _ = np.histogram(lbp_masked, bins=np.arange(0, 11), density=True)
            lbp_uniformity = lbp_hist[0]  # Uniform patterns proportion
        except Exception:
            lbp_uniformity = 0

        features.append({
            'area': area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'compactness': compactness,
            'circularity': circularity,
            'centroid': (cx, cy),
            'contour': contour,
            'fill_ratio': fill_ratio,
            'defects_count': defects_count,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'lbp_uniformity': lbp_uniformity,
            'norm_area': norm_area,
            'norm_perimeter': norm_perimeter
        })
    return features

def classify_pore_vs_blowhole(roi, original_class):
    # Removed DEMO OVERRIDE: No longer force classification as blowhole for large central ROI
    """Classify whether a detection is porosity or blowhole based on geometric features"""
    features = analyze_contour_features(roi)

    if not features:
        print("    [DEBUG] No features extracted. Returning original class.")
        return original_class, 0.7


    # Count significant contours (area > 20)
    significant_contours = [f for f in features if f['area'] > 20]
    total_area = sum(f['area'] for f in significant_contours)
    centroids = np.array([f['centroid'] for f in significant_contours]) if significant_contours else np.empty((0,2))

    print(f"    [DEBUG] Found {len(significant_contours)} significant contours, total area: {total_area}")
    for idx, f in enumerate(significant_contours):
        print(f"      [DEBUG] Contour {idx+1}: area={f['area']:.1f}, norm_area={f['norm_area']:.4f}, aspect_ratio={f['aspect_ratio']:.2f}, solidity={f['solidity']:.2f}, compactness={f['compactness']:.2f}, circularity={f['circularity']:.2f}, fill_ratio={f['fill_ratio']:.2f}, defects_count={f['defects_count']}, mean_intensity={f['mean_intensity']:.1f}, std_intensity={f['std_intensity']:.1f}, lbp_uniformity={f['lbp_uniformity']:.2f}")

    # --- FORCE BLOWHOLE RULE: If ANY significant contour is very large, round, and solid, always classify as blowhole ---
    for f in significant_contours:
        if (
            f['norm_area'] > 0.28 and  # Lowered threshold to catch more large blowholes
            f['solidity'] > 0.75 and
            f['circularity'] > 0.8 and
            f['fill_ratio'] > 0.7 and
            f['defects_count'] <= 2 and
            f['std_intensity'] < 45
        ):
            print("    [FORCE RULE] Large, round, solid contour detected (area={:.2f}, norm_area={:.3f}). Forcing classification as blowhole.".format(f['area'], f['norm_area']))
            return 'blowholes', 0.995

    # Cluster analysis: DBSCAN on centroids
    cluster_count = 0
    if len(centroids) > 1:
        try:
            db = DBSCAN(eps=20, min_samples=2).fit(centroids)
            cluster_count = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        except Exception:
            cluster_count = 0

    # Use classifier if available and at least one significant contour
    if clf is not None and scaler is not None and label_encoder is not None and significant_contours:
        main_contour = max(significant_contours, key=lambda x: x['area'])
        feature_vector = [
            main_contour['norm_area'], main_contour['aspect_ratio'], main_contour['solidity'], main_contour['compactness'],
            main_contour['circularity'], main_contour['fill_ratio'], main_contour['defects_count'],
            main_contour['mean_intensity'], main_contour['std_intensity'], main_contour['lbp_uniformity']
        ]
        # Scale the feature vector
        feature_vector_scaled = scaler.transform([feature_vector])
        clf_pred_encoded = clf.predict(feature_vector_scaled)[0]
        clf_pred = label_encoder.inverse_transform([clf_pred_encoded])[0]
        print(f"    [DEBUG] Classifier prediction: {clf_pred}")
    else:
        clf_pred = None

    # Tighter rules for blowhole/porosity distinction (now using normalized area)
    if original_class == 'blowholes':
        if significant_contours:
            main_contour = max(significant_contours, key=lambda x: x['area'])
            print(f"    [DEBUG] Blowhole analysis: main area={main_contour['area']:.1f}, norm_area={main_contour['norm_area']:.4f}, solidity={main_contour['solidity']:.2f}, circularity={main_contour['circularity']:.2f}, fill_ratio={main_contour['fill_ratio']:.2f}, defects={main_contour['defects_count']}, mean_int={main_contour['mean_intensity']:.1f}, std_int={main_contour['std_intensity']:.1f}, lbp_uniformity={main_contour['lbp_uniformity']:.2f}")

            rule_blowhole = (
                main_contour['norm_area'] > 0.25 and main_contour['solidity'] > 0.7 and main_contour['circularity'] > 0.8 and
                main_contour['fill_ratio'] > 0.7 and main_contour['defects_count'] <= 2 and main_contour['std_intensity'] < 35 and main_contour['lbp_uniformity'] > 0.25
            )
            if rule_blowhole and clf_pred == 'blowhole':
                return 'blowholes', 0.98
            elif rule_blowhole or clf_pred == 'blowhole':
                return 'blowholes', 0.9
            elif main_contour['norm_area'] > 0.15 and main_contour['solidity'] > 0.6:
                return 'blowholes', 0.8
            elif main_contour['norm_area'] > 0.08:
                return 'blowholes', 0.7
        return 'porosity', 0.6

    # For porosity detections, look for clustering patterns
    else:  # original_class == 'porosity'
        if len(significant_contours) >= 3 or cluster_count > 1:
            if clf_pred == 'porosity':
                return 'porosity', 0.99
            else:
                return 'porosity', 0.97

        if significant_contours:
            main_contour = max(significant_contours, key=lambda x: x['area'])
            print(f"    [DEBUG] Porosity analysis: main area={main_contour['area']:.1f}, norm_area={main_contour['norm_area']:.4f}, solidity={main_contour['solidity']:.2f}, circularity={main_contour['circularity']:.2f}, fill_ratio={main_contour['fill_ratio']:.2f}, defects={main_contour['defects_count']}, mean_int={main_contour['mean_intensity']:.1f}, std_int={main_contour['std_intensity']:.1f}, lbp_uniformity={main_contour['lbp_uniformity']:.2f}")

            rule_blowhole = (
                len(significant_contours) == 1 and main_contour['norm_area'] > 0.25 and main_contour['solidity'] > 0.7 and
                main_contour['circularity'] > 0.8 and main_contour['fill_ratio'] > 0.7 and main_contour['defects_count'] <= 2 and main_contour['std_intensity'] < 35 and main_contour['lbp_uniformity'] > 0.25
            )
            if rule_blowhole and clf_pred == 'blowhole':
                return 'blowholes', 0.95
            elif rule_blowhole or clf_pred == 'blowhole':
                return 'blowholes', 0.9
            elif len(significant_contours) == 1 and main_contour['norm_area'] > 0.25 and main_contour['solidity'] > 0.7:
                return 'blowholes', 0.8
            else:
                if clf_pred == 'porosity':
                    return 'porosity', 0.92
                else:
                    return 'porosity', 0.88

        return 'porosity', 0.7

def apply_nms_to_pore_detections(detections, iou_threshold=0.3):
    """Apply Non-Maximum Suppression to pore detections to remove overlaps"""
    if not detections:
        return detections
    
    # Sort by confidence
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while sorted_detections:
        current = sorted_detections.pop(0)
        keep.append(current)
        
        # Remove overlapping detections
        remaining = []
        for det in sorted_detections:
            iou = calculate_iou(current['bbox'], det['bbox'])
            if iou < iou_threshold:
                remaining.append(det)
            else:
                print(f"  NMS: Removing overlapping {det['type']} (IoU={iou:.2f} with {current['type']})")
        
        sorted_detections = remaining
    
    return keep

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def detect_and_classify_pores(image, models, confidence_thresholds):
    """Detect and classify porosity vs blowholes using geometric analysis"""
    # Combine detections from both porosity and blowhole models
    pore_boxes, pore_confs = detect_with_model(models['porosity'], image, confidence_thresholds['porosity'])
    blow_boxes, blow_confs = detect_with_model(models['blowholes'], image, confidence_thresholds['blowholes'])
    
    print(f"  Raw YOLO detections: {len(pore_boxes)} porosity, {len(blow_boxes)} blowholes")
    
    all_detections = []
    
    # Process porosity detections
    for i, (box, conf) in enumerate(zip(pore_boxes, pore_confs)):
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]
        
        print(f"  Processing porosity detection {i+1}: confidence={conf:.2f}")
        classified_type, geo_conf = classify_pore_vs_blowhole(roi, 'porosity')
        combined_conf = (conf * 0.6) + (geo_conf * 0.4)  # Weight YOLO confidence more
        
        all_detections.append({
            'bbox': box,
            'type': classified_type,
            'confidence': combined_conf,
            'original_model': 'porosity'
        })
    
    # Process blowhole detections
    for i, (box, conf) in enumerate(zip(blow_boxes, blow_confs)):
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]
        
        print(f"  Processing blowhole detection {i+1}: confidence={conf:.2f}")
        classified_type, geo_conf = classify_pore_vs_blowhole(roi, 'blowholes')
        combined_conf = (conf * 0.6) + (geo_conf * 0.4)
        
        all_detections.append({
            'bbox': box,
            'type': classified_type,
            'confidence': combined_conf,
            'original_model': 'blowholes'
        })
    
    # Suppress porosity detections that overlap with blowhole detections
    final_detections = []
    blowhole_boxes = [d['bbox'] for d in all_detections if d['type'] == 'blowholes']
    for det in all_detections:
        if det['type'] == 'porosity':
            overlap = False
            for bh_box in blowhole_boxes:
                # Calculate IoU between porosity and blowhole box
                iou = calculate_iou(det['bbox'], bh_box)
                if iou > 0.3:  # You can adjust this threshold as needed
                    overlap = True
                    break
            if not overlap:
                final_detections.append(det)
        else:
            final_detections.append(det)
    return final_detections

def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return

    # 1. Detect weldingPlate

    plate_boxes, plate_confs = detect_with_model(models['weldingPlate'], image, confidence_thresholds['weldingPlate'])
    if len(plate_boxes) == 0:
        print("No welding plate detected.")
    else:
        print(f"Welding plate detected: {len(plate_boxes)} region(s)")
        # Draw plate boxes
        for box, conf in zip(plate_boxes, plate_confs):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'weldingPlate ({conf:.2f})', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # 2. Detect weld_seam
    seam_boxes, seam_confs = detect_with_model(models['weld_seam'], image, confidence_thresholds['weld_seam'])
    if len(seam_boxes) == 0:
        print("No weld seam detected.")
    else:
        print(f"Weld seam detected: {len(seam_boxes)} region(s)")
        for box, conf in zip(seam_boxes, seam_confs):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f'weld_seam ({conf:.2f})', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # 3. Detect and classify porosity/blowholes using geometric analysis
    print("Detecting and classifying porosity/blowholes...")
    pore_detections = detect_and_classify_pores(image, models, confidence_thresholds)
    
    # Apply NMS to remove overlapping detections
    print("Applying NMS to remove overlapping detections...")
    pore_detections = apply_nms_to_pore_detections(pore_detections, iou_threshold=0.5)  # Higher threshold to be less aggressive
    
    # Count and display results
    porosity_count = len([d for d in pore_detections if d['type'] == 'porosity'])
    blowhole_count = len([d for d in pore_detections if d['type'] == 'blowholes'])
    print(f"Porosity detected: {porosity_count} region(s)")
    print(f"Blowholes detected: {blowhole_count} region(s)")
    
    # Draw pore detections with geometric classification
    defect_colors = {
        'porosity': (255, 0, 255),    # Magenta/Pink in BGR format - very distinct
        'blowholes': (0, 255, 0),     # Green in BGR format - very distinct
    }
    
    for detection in pore_detections:
        box = detection['bbox']
        defect_type = detection['type']
        confidence = detection['confidence']
        x1, y1, x2, y2 = map(int, box)
        color = defect_colors.get(defect_type, (0, 0, 0))
        
        # Debug output for each detection
        print(f"  Drawing {defect_type} at ({x1},{y1})-({x2},{y2}) with color {color} and confidence {confidence:.2f}")
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)  # Thicker border
        cv2.putText(image, f'{defect_type} ({confidence:.2f})', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4. Detect other defects (with suppression for excessive reinforcement)
    other_defect_classes = ['excessive_reinforcement', 'crack', 'spatter']
    other_defect_colors = {
        'excessive_reinforcement': (255, 0, 255),
        'crack': (0, 0, 255),
        'spatter': (128, 128, 128)
    }
    # Collect blowhole and porosity boxes for suppression
    suppress_boxes = [d['bbox'] for d in pore_detections if d['type'] in ['blowholes', 'porosity']]
    for defect in other_defect_classes:
        boxes, confs = detect_with_model(models[defect], image, confidence_thresholds[defect])
        print(f"{defect.capitalize()} detected: {len(boxes)} region(s)")
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            color = other_defect_colors.get(defect, (0, 0, 0))
            # Suppress excessive_reinforcement if it overlaps with blowhole or porosity
            if defect == 'excessive_reinforcement':
                overlap = False
                for sbox in suppress_boxes:
                    iou = calculate_iou(box, sbox)
                    if iou > 0.3:
                        overlap = True
                        break
                if overlap:
                    print(f"Suppressed excessive_reinforcement at ({x1},{y1})-({x2},{y2}) due to overlap with blowhole/porosity.")
                    continue
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f'{defect} ({conf:.2f})', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the output image
    output_path = 'output_with_detections.jpg'
    cv2.imwrite(output_path, image)
    print(f"Output image saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = r"C:\Users\debra\Downloads\why_do_i_blow_holes_when_mig_welding_1-1800x800_1800_x_840_px_813acc06-bbcb-4cc8-aaa2-d63ca451e7b2_2000x.JPG"
    main(image_path)
