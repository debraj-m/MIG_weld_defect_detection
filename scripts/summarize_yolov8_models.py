import os
from ultralytics import YOLO

# List of YOLOv8 .pt files to summarize
MODEL_FILES = [
    'weights/blowholes_detect.pt',
    'weights/crack_defect.pt',
    'weights/pore_detect.pt',
    'weights/excessive_reinforcement.pt',
    'weights/spatter_defect.pt',
]

def summarize_model(model_path):
    print(f'\n===== Model: {os.path.basename(model_path)} =====')
    try:
        model = YOLO(model_path)
        print('Model type:', type(model.model).__name__)
        print('Number of parameters:', sum(p.numel() for p in model.model.parameters()))
        if hasattr(model.model, 'names'):
            print('Classes:', model.model.names)
        if hasattr(model.model, 'yaml'):
            print('Model YAML:', model.model.yaml)
        if hasattr(model.model, 'nc'):
            print('Number of classes:', model.model.nc)
        # Print model structure summary
        print('\nModel structure:')
        print(model.model)
    except Exception as e:
        print(f'Error loading {model_path}: {e}')

def main():
    for model_file in MODEL_FILES:
        if os.path.exists(model_file):
            summarize_model(model_file)
        else:
            print(f'File not found: {model_file}')

if __name__ == '__main__':
    main()
