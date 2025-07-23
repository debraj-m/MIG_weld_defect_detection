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
    'porosity': 0.1,
    'blowholes': 0.5,
    'excessive_reinforcement': 0.4,
    'crack': 0.4,
    'spatter': 0.3
}

# Paths for classifier, scaler, and label encoder
models_dir = os.path.abspath(os.path.join(BASE_DIR, '../models'))
classifier_path = os.path.join(models_dir, 'defect_classifier.joblib')
scaler_path = os.path.join(models_dir, 'defect_scaler.joblib')
label_encoder_path = os.path.join(models_dir, 'defect_label_encoder.joblib')
