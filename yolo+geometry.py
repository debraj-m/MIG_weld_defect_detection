import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import json
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import os
from feature_stats import GeometricFeatureStats

class IntegratedWeldDefectDetector:
    def __init__(self, weldseam_model_path: str, porosity_model_path: str, crack_model_path: str):
        """
        Initialize the integrated detector with pre-trained models
        
        Args:
            weldseam_model_path: Path to weldseam.pt model
            porosity_model_path: Path to porosity.pt model  
            crack_model_path: Path to crack.pt model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained models
        self.weldseam_model = self._load_model(weldseam_model_path)
        self.porosity_model = self._load_model(porosity_model_path)
        self.crack_model = self._load_model(crack_model_path)
        
        # Image preprocessing for models
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),  # Adjust based on your model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Contour detector for geometric validation
        self.contour_detector = WeldContourDetector()
        
        # Load geometric feature statistics for data-driven validation
        porosity_stats_path = os.path.join(os.path.dirname(__file__), 'porosity_geometric_features.csv')
        crack_stats_path = os.path.join(os.path.dirname(__file__), 'crack_geometric_features.csv')
        self.feature_stats_porosity = GeometricFeatureStats(porosity_stats_path)
        self.feature_stats_crack = GeometricFeatureStats(crack_stats_path)
        
        # Detection results
        self.weld_regions = []
        self.defect_detections = []
        self.validated_defects = []
        
    def _load_model(self, model_path: str):
        """Load YOLO model from .pt file using Ultralytics API"""
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None
    
    def detect_weld_seam(self, image: np.ndarray) -> List[Dict]:
        """
        Detect weld seam using weldseam.pt model
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of weld seam detections with bounding boxes
        """
        if self.weldseam_model is None:
            print("Weld seam model not loaded")
            return []
        
        # Pass the raw image (numpy array) directly to YOLO
        results = self.weldseam_model(image)
        weld_regions = self._process_weldseam_predictions(results[0], image.shape)
        self.weld_regions = weld_regions
        
        return weld_regions
    
    def detect_defects_in_regions(self, image: np.ndarray, weld_regions: List[Dict]) -> List[Dict]:
        """
        Detect defects within weld seam regions using porosity and crack models
        
        Args:
            image: Input image
            weld_regions: List of weld seam regions from detect_weld_seam
            
        Returns:
            List of defect detections
        """
        all_defects = []
        
        for region in weld_regions:
            # Extract ROI from weld region
            roi = self._extract_roi(image, region)
            
            # Detect porosity
            porosity_detections = self._detect_porosity(roi, region)
            
            # Detect cracks
            crack_detections = self._detect_cracks(roi, region)
            
            # Combine detections
            region_defects = porosity_detections + crack_detections
            all_defects.extend(region_defects)
        
        self.defect_detections = all_defects
        return all_defects
    
    def _detect_porosity(self, roi: np.ndarray, region: Dict) -> List[Dict]:
        """Detect porosity using porosity.pt model"""
        if self.porosity_model is None:
            return []
        
        # Pass the raw ROI (numpy array) directly to YOLO
        results = self.porosity_model(roi)
        porosity_detections = self._process_defect_predictions(
            results[0], region, 'porosity', roi.shape
        )
        return porosity_detections

    def _detect_cracks(self, roi: np.ndarray, region: Dict) -> List[Dict]:
        """Detect cracks using crack.pt model"""
        if self.crack_model is None:
            return []
        
        # Pass the raw ROI (numpy array) directly to YOLO
        results = self.crack_model(roi)
        crack_detections = self._process_defect_predictions(
            results[0], region, 'crack', roi.shape
        )
        return crack_detections
    
    def validate_with_contour_analysis(self, image: np.ndarray, defect_detections: List[Dict]) -> List[Dict]:
        """
        Validate defect detections using contour analysis
        
        Args:
            image: Original image
            defect_detections: Raw defect detections from models
            
        Returns:
            Validated defect detections
        """
        validated_defects = []
        
        for detection in defect_detections:
            # Extract defect region
            defect_roi = self._extract_roi(image, detection)
            
            # Apply contour analysis
            processed = self.contour_detector.preprocess_image(defect_roi)
            edges = self.contour_detector.detect_edges(processed, method='canny')
            contours = self.contour_detector.extract_contours(edges)
            
            if contours:
                # Analyze contour features
                features = self.contour_detector.analyze_contour_features(contours, defect_roi)
                
                # Validate detection based on geometric features
                validation_result = self._validate_detection(detection, features)
                print(f"Validation for {detection['type']} (conf: {detection['confidence']:.2f}): {validation_result['validation_reason']} | Valid: {validation_result['is_valid']}")
                if validation_result['is_valid']:
                    # Enhance detection with geometric features
                    enhanced_detection = detection.copy()
                    enhanced_detection.update(validation_result)
                    validated_defects.append(enhanced_detection)
            else:
                print(f"Validation for {detection['type']} (conf: {detection['confidence']:.2f}): No contours found | Valid: False")
        
        self.validated_defects = validated_defects
        return validated_defects
    
    def _validate_detection(self, detection: Dict, contour_features: List[Dict]) -> Dict:
        """
        Validate a single detection using contour features
        
        Args:
            detection: Defect detection from model
            contour_features: Geometric features from contour analysis
            
        Returns:
            Validation result with confidence and geometric metrics
        """
        if not contour_features:
            return {'is_valid': False, 'confidence': 0.0, 'reason': 'No contours found', 'validation_reason': 'No contours found'}
        
        # Get the largest contour (most likely the defect)
        main_feature = max(contour_features, key=lambda x: x['area'])
        
        defect_type = detection['type']
        model_confidence = detection['confidence']
        
        # Use the correct stats for each defect type
        if defect_type == 'porosity':
            stats = self.feature_stats_porosity
        elif defect_type == 'crack':
            stats = self.feature_stats_crack
        else:
            stats = self.feature_stats_porosity  # fallback
        
        # Data-driven validation using feature stats
        if defect_type == 'porosity':
            in_range = (
                stats.is_within_range('area', main_feature['area']) and
                stats.is_within_range('aspect_ratio', main_feature['aspect_ratio']) and
                stats.is_within_range('solidity', main_feature['solidity']) and
                stats.is_within_range('compactness', main_feature['compactness'])
            )
            is_valid = in_range
            # Confidence boost if features are close to mean
            geom_score = 1 - (abs(main_feature['area'] - stats.get_stats('area')['mean']) / (stats.get_stats('area')['std'] + 1e-6))
            geom_score = max(0, min(geom_score, 1))
            geometric_confidence = min(main_feature['solidity'], 1.0) * geom_score
        elif defect_type == 'crack':
            # Use aspect_ratio and compactness for cracks
            aspect_ratio = main_feature.get('aspect_ratio', 0)
            compactness = main_feature.get('compactness', 0)
            # Robustness: handle NaN/None/extreme values
            if aspect_ratio is None or not np.isfinite(aspect_ratio):
                aspect_ratio = 0
            if compactness is None or not np.isfinite(compactness):
                compactness = 0
            # Fallback for very small or irregular contours
            if main_feature.get('area', 0) < 10 or aspect_ratio < 1e-3:
                is_valid = False
                geometric_confidence = 0.0
                validation_reason = 'Contour too small or aspect_ratio too low'
            else:
                # Use median/IQR if available, else mean/std
                stats_aspect = stats.get_stats('aspect_ratio')
                stats_compact = stats.get_stats('compactness')
                aspect_center = stats_aspect.get('median', stats_aspect.get('mean', 0))
                aspect_spread = stats_aspect.get('iqr', stats_aspect.get('std', 1e-6))
                compact_center = stats_compact.get('median', stats_compact.get('mean', 0))
                compact_spread = stats_compact.get('iqr', stats_compact.get('std', 1e-6))
                # Parameterize divisor and epsilon
                aspect_divisor = 10.0
                epsilon = 1e-6
                in_range = (
                    stats.is_within_range('aspect_ratio', aspect_ratio) and
                    stats.is_within_range('compactness', compactness)
                )
                is_valid = in_range
                # Robust geometric score
                geom_score = 1 - (abs(aspect_ratio - aspect_center) / (aspect_spread + epsilon))
                geom_score = max(0, min(geom_score, 1))
                geometric_confidence = min(aspect_ratio / aspect_divisor, 1.0) * geom_score
                validation_reason = (
                    f"Area: {main_feature.get('area', 0):.2f} (in range: {stats.is_within_range('area', main_feature.get('area', 0))}), "
                    f"Aspect ratio: {aspect_ratio:.2f} (in range: {stats.is_within_range('aspect_ratio', aspect_ratio)}), "
                    f"Compactness: {compactness:.2f} (in range: {stats.is_within_range('compactness', compactness)})"
                )
        else:
            is_valid = True
            geometric_confidence = 0.5
            validation_reason = 'Default/fallback case'
        # Combined confidence score
        combined_confidence = (model_confidence * 0.6) + (geometric_confidence * 0.4)
        
        return {
            'is_valid': is_valid,
            'confidence': combined_confidence,
            'geometric_confidence': geometric_confidence,
            'model_confidence': model_confidence,
            'geometric_features': main_feature,
            'validation_reason': validation_reason
        }
    
    def ensemble_prediction(self, image: np.ndarray, apply_nms: bool = True, use_geometric_validation: bool = True) -> Dict:
        """
        Complete ensemble prediction pipeline
        
        Args:
            image: Input image
            apply_nms: Whether to apply non-maximum suppression
            use_geometric_validation: Whether to use geometric validation (contour analysis)
        Returns:
            Complete detection results
        """
        # Step 1: Detect weld seams
        print("Detecting weld seams...")
        weld_regions = self.detect_weld_seam(image)

        # Fallback: If no weld seams found, treat the whole image as one region
        if not weld_regions:
            print("No weld seams found, running defect detection on the whole image.")
            h, w = image.shape[:2]
            weld_regions = [{
                'bbox': [0, 0, w, h],
                'confidence': 1.0,
                'type': 'weld_seam',
                'id': 0
            }]

        # Step 2: Detect defects in weld regions
        print("Detecting defects in weld regions...")
        defect_detections = self.detect_defects_in_regions(image, weld_regions)
        
        # Step 3: Optionally validate with contour analysis
        if use_geometric_validation:
            print("Validating detections with contour analysis...")
            validated_defects = self.validate_with_contour_analysis(image, defect_detections)
        else:
            validated_defects = defect_detections
        
        # Step 4: Apply non-maximum suppression if requested
        if apply_nms:
            validated_defects = self._apply_nms(validated_defects)
        
        # Step 5: Final filtering based on combined confidence
        final_defects = [d for d in validated_defects if d['confidence'] > 0.5]
        
        return {
            'weld_regions': weld_regions,
            'raw_defects': defect_detections,
            'validated_defects': final_defects,
            'summary': self._generate_summary(final_defects)
        }
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return detections
        
        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while sorted_detections:
            current = sorted_detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            sorted_detections = [
                det for det in sorted_detections 
                if self._calculate_iou(current, det) < iou_threshold
            ]
        
        return keep
    
    def _calculate_iou(self, det1: Dict, det2: Dict) -> float:
        """Calculate Intersection over Union between two detections"""
        # Extract bounding boxes
        x1_1, y1_1, x2_1, y2_1 = det1['bbox']
        x1_2, y1_2, x2_2, y2_2 = det2['bbox']
        
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
    
    def visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Visualize detection results on the image
        
        Args:
            image: Original image
            results: Detection results from ensemble_prediction
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Draw weld seam regions
        for region in results['weld_regions']:
            x1, y1, x2, y2 = [int(round(v)) for v in region['bbox']]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow
            cv2.putText(annotated, 'Weld Seam', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw validated defects
        colors = {'porosity': (0, 255, 0), 'crack': (0, 0, 255)}  # Green, Red
        
        for defect in results['validated_defects']:
            x1, y1, x2, y2 = [int(round(v)) for v in defect['bbox']]
            color = colors.get(defect['type'], (128, 128, 128))
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{defect['type']} ({defect['confidence']:.2f})"
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
    
    def _generate_summary(self, defects: List[Dict]) -> Dict:
        """Generate summary statistics"""
        summary = {
            'total_defects': len(defects),
            'porosity_count': len([d for d in defects if d['type'] == 'porosity']),
            'crack_count': len([d for d in defects if d['type'] == 'crack']),
            'avg_confidence': np.mean([d['confidence'] for d in defects]) if defects else 0.0,
            'high_confidence_defects': len([d for d in defects if d['confidence'] > 0.8])
        }
        return summary
    
    # Helper methods for processing model outputs (customize based on your model formats)
    def _process_weldseam_predictions(self, predictions, image_shape):
        """Process weldseam model predictions - customize based on your model output"""
        detections = []
        if hasattr(predictions, 'boxes'):
            # Use YOLO's .boxes.xyxy and .boxes.conf for bounding boxes and confidence
            boxes = predictions.boxes.xyxy.cpu().numpy()
            scores = predictions.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                box = [float(x) for x in box]
                if len(box) == 4 and score > 0.5:
                    detections.append({
                        'bbox': box,
                        'confidence': float(score),
                        'type': 'weld_seam'
                    })
        return detections

    def _process_defect_predictions(self, predictions, region, defect_type, roi_shape):
        """Process defect model predictions - customize based on your model output"""
        detections = []
        if hasattr(predictions, 'boxes'):
            boxes = predictions.boxes.xyxy.cpu().numpy()
            scores = predictions.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                box = [float(x) for x in box]
                if len(box) == 4 and score > 0.3:
                    global_box = self._roi_to_global_coords(box, region)
                    detections.append({
                        'bbox': global_box,
                        'confidence': float(score),
                        'type': defect_type,
                        'region_id': region.get('id', 0)
                    })
        return detections
    
    def _extract_roi(self, image, detection):
        """Extract region of interest from image based on detection bbox"""
        x1, y1, x2, y2 = detection['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        return image[y1:y2, x1:x2]
    
    def _roi_to_global_coords(self, roi_box, region):
        """Convert ROI coordinates to global image coordinates"""
        region_x1, region_y1, _, _ = region['bbox']
        
        if isinstance(roi_box, torch.Tensor):
            roi_box = roi_box.tolist()
        
        x1, y1, x2, y2 = roi_box
        return [x1 + region_x1, y1 + region_y1, x2 + region_x1, y2 + region_y1]


class WeldContourDetector:
    """Simplified contour detector for integration"""
    def __init__(self):
        self.processed_image = None
        self.edges = None
        self.contours = []
    
    def preprocess_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        enhanced = cv2.equalizeHist(denoised)
        smoothed = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        self.processed_image = smoothed
        return smoothed
    
    def detect_edges(self, image, method='canny'):
        edges = cv2.Canny(image, 50, 150, apertureSize=3, L2gradient=True)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        self.edges = edges
        return edges
    
    def extract_contours(self, edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > 50]
        self.contours = filtered_contours
        return filtered_contours
    
    def analyze_contour_features(self, contours, original_image):
        features = []
        
        for i, contour in enumerate(contours):
            if len(contour) < 5:
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
            }
            
            features.append(feature)
        
        return features


def main():
    """Example usage of the integrated system"""
    # Initialize the integrated detector
    detector = IntegratedWeldDefectDetector(
        weldseam_model_path=r'weights\weld_seam.pt',
        porosity_model_path=r'weights\pore_detect.pt', 
        crack_model_path=r'weights\crack_defect.pt'
    )
    
    # Load your image
    image = cv2.imread(r"C:\Users\debra\Desktop\CODE\Dataset\Datasets\crack_dataset\train\images\15_jpeg_jpg.rf.ebdff550f0f589b1429574e4207503ea.jpg")
    
    # Run complete detection pipeline (set use_geometric_validation=False for raw YOLO results)
    results = detector.ensemble_prediction(image, use_geometric_validation=False)
    
    # Visualize results
    annotated_image = detector.visualize_results(image, results)
    
    # Save annotated image to outputs directory
    import os
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'annotated_output.jpg')
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved to: {output_path}")
    
    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title('Detection Results')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n=== DETECTION SUMMARY ===")
    print(f"Weld regions found: {len(results['weld_regions'])}")
    print(f"Total defects detected: {results['summary']['total_defects']}")
    print(f"Porosity defects: {results['summary']['porosity_count']}")
    print(f"Crack defects: {results['summary']['crack_count']}")
    print(f"Average confidence: {results['summary']['avg_confidence']:.2f}")
    print(f"High confidence defects: {results['summary']['high_confidence_defects']}")

if __name__ == "__main__":
    main()