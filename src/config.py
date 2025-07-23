# config.py
# Centralized configuration for model weights, thresholds, and ML assets

# Paths to weights (update as needed)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.abspath(os.path.join(BASE_DIR, '../weights'))
weights = {
    'weldingPlate': os.path.join(weights_dir, 'weldingPlate.pt'),
    'weld_seam': os.path.join(weights_dir, 'weld_seam.pt'),
    'porosity': os.path.join(weights_dir, 'pore_detect.pt'),
    'blowholes': os.path.join(weights_dir, 'blowholes_detect.pt'),
    'excessive_reinforcement': os.path.join(weights_dir, 'excessive_reinforcement.pt'),
    'crack': os.path.join(weights_dir, 'crack_defect.pt'),
    'spatter': os.path.join(weights_dir, 'spatter_defect.pt'),
}

# Confidence thresholds for each model
confidence_thresholds = {
    'weldingPlate': 0.5,
    'weld_seam': 0.5,
    'porosity': 0.1,        # Lower threshold for small defects
    'blowholes': 0.5,
    'excessive_reinforcement': 0.4,
    'crack': 0.4,
    'spatter': 0.3
}

# Overlap threshold for filtering similar detections
overlap_threshold = 0.5

# Minimum defect area (in pixels) for filtering noise
min_defect_area = 50

# Feature extraction parameters
feature_extraction_params = {
    'gaussian_blur_kernel': (5, 5),
    'canny_low_threshold': 50,
    'canny_high_threshold': 150,
    'morphology_kernel_size': (3, 3),
    'min_contour_area': 10,
    'bilateral_filter_d': 9,
    'bilateral_filter_sigma_color': 75,
    'bilateral_filter_sigma_space': 75
}

# Classification threshold for porosity vs blowhole
classification_confidence_threshold = 0.7

# Paths for classifier, scaler, and label encoder
models_dir = os.path.abspath(os.path.join(BASE_DIR, '../models'))
classifier_path = os.path.join(models_dir, 'defect_classifier.joblib')
scaler_path = os.path.join(models_dir, 'defect_scaler.joblib')
label_encoder_path = os.path.join(models_dir, 'defect_label_encoder.joblib')

# Results and logging configuration
results_dir = os.path.abspath(os.path.join(BASE_DIR, '../results'))
log_level = 'INFO'  # DEBUG, INFO, WARNING, ERROR

# Performance monitoring
performance_metrics = {
    'track_inference_time': True,
    'track_memory_usage': True,
    'save_detection_results': True,
    'generate_performance_reports': True
}
