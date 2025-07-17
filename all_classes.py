from ultralytics import YOLO
import cv2
import sys
import numpy as np
from sklearn.cluster import DBSCAN

image_path= r"C:\Users\debra\Desktop\CODE\Dataset\Test_images\Experiment Photos\PXL_20250525_104231099.jpg"

# Paths to weights (update if needed)
weights_dir = r'c:\Users\debra\Desktop\CODE\Dataset\weights'
models = {
    'weldingPlate': f'{weights_dir}\\weldingPlate.pt',
    'weld_seam': f'{weights_dir}\\weld_seam.pt',
    'porosity': f'{weights_dir}\\pore_detect.pt',
    'blowholes': f'{weights_dir}\\blowholes_detect.pt',
    'excessive_reinforcement': f'{weights_dir}\\excessive_reinforcement.pt',
    'crack': f'{weights_dir}\\crack_defect.pt',
    'spatter': f'{weights_dir}\\spatter_defect.pt',  # Update if needed
}

# Confidence thresholds for each model
confidence_thresholds = {
    'weldingPlate': 0.5,
    'weld_seam': 0.5,
    'porosity': 0.4,
    'blowholes': 0.4,
    'excessive_reinforcement': 0.4,
    'crack': 0.4,
    'spatter': 0.3  # Lower threshold for spatter detection
}

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
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        if area < 10:  # Skip very small contours
            continue
            
        # Calculate geometric features
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Aspect ratio
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if height > 0:
            aspect_ratio = width / height
        else:
            aspect_ratio = 1.0
            
        # Solidity
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Compactness
        compactness = (perimeter * perimeter) / area if area > 0 else 0
        
        # Centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
            
        features.append({
            'area': area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'compactness': compactness,
            'centroid': (cx, cy),
            'contour': contour
        })
    
    return features

def classify_pore_vs_blowhole(roi, original_class):
    """Classify whether a detection is porosity or blowhole based on geometric features"""
    features = analyze_contour_features(roi)
    
    if not features:
        return original_class, 0.7  # Higher confidence for original if no features
    
    # Count significant contours (area > 15)
    significant_contours = [f for f in features if f['area'] > 15]
    total_area = sum(f['area'] for f in significant_contours)
    
    print(f"  Debug: Found {len(significant_contours)} significant contours, total area: {total_area}")
    
    # For blowhole detections, be more lenient - focus on the main contour
    if original_class == 'blowholes':
        if significant_contours:
            # Find the largest contour (main defect)
            main_contour = max(significant_contours, key=lambda x: x['area'])
            print(f"  -> Blowhole analysis: main area={main_contour['area']:.1f}, solidity={main_contour['solidity']:.2f}")
            
            # If main contour is reasonably large and well-formed, keep as blowhole
            if main_contour['area'] > 25 and main_contour['solidity'] > 0.3:
                print(f"  -> Good main contour: Keeping as BLOWHOLE")
                return 'blowholes', 0.8
            # Even if not perfect, respect YOLO's blowhole detection for medium-sized defects
            elif main_contour['area'] > 15:
                print(f"  -> Medium defect: Keeping as BLOWHOLE (trust YOLO)")
                return 'blowholes', 0.7
        
        print(f"  -> Weak blowhole features: Reclassifying as POROSITY")
        return 'porosity', 0.6
    
    # For porosity detections, look for clustering patterns
    else:  # original_class == 'porosity'
        # Multiple small contours strongly suggest porosity cluster
        if len(significant_contours) >= 3:
            print(f"  -> Multiple contours: Confirmed POROSITY")
            return 'porosity', 0.9
        
        # Single or two contours - check size and shape
        if len(significant_contours) >= 1:
            main_contour = max(significant_contours, key=lambda x: x['area'])
            print(f"  -> Porosity analysis: main area={main_contour['area']:.1f}, solidity={main_contour['solidity']:.2f}")
            
            # Large, well-formed single contour might be a blowhole
            if len(significant_contours) == 1 and main_contour['area'] > 80 and main_contour['solidity'] > 0.6:
                print(f"  -> Large solid contour: Reclassifying as BLOWHOLE")
                return 'blowholes', 0.75
            else:
                print(f"  -> Confirmed POROSITY")
                return 'porosity', 0.8
        
        print(f"  -> Default: Keeping as POROSITY")
        return 'porosity', 0.6

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
    
    return all_detections

def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return

    # 1. Detect weldingPlate
    plate_boxes, plate_confs = detect_with_model(models['weldingPlate'], image, confidence_thresholds['weldingPlate'])
    if len(plate_boxes) == 0:
        print("No welding plate detected.")
        return
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
        return
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

    # 4. Detect other defects (without geometric analysis)
    other_defect_classes = ['excessive_reinforcement', 'crack', 'spatter']
    other_defect_colors = {
        'excessive_reinforcement': (255, 0, 255),
        'crack': (0, 0, 255),
        'spatter': (128, 128, 128)
    }
    for defect in other_defect_classes:
        boxes, confs = detect_with_model(models[defect], image, confidence_thresholds[defect])
        print(f"{defect.capitalize()} detected: {len(boxes)} region(s)")
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            color = other_defect_colors.get(defect, (0, 0, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f'{defect} ({conf:.2f})', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the output image
    output_path = 'output_with_detections.jpg'
    cv2.imwrite(output_path, image)
    print(f"Output image saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    main(image_path)
