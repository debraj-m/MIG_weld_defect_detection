# MIGWeld Defect Detection

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An advanced computer vision pipeline for automated defect detection in Metal Inert Gas (MIG) welding processes using state-of-the-art YOLOv8 object detection. This project enables real-time quality control and inspection in industrial welding applications.

## 🎯 Overview

This repository contains a complete object detection solution developed during an AI/ML internship to support automated quality control in industrial welding processes. The system can identify and classify various types of welding defects that are critical for ensuring structural integrity and safety standards.

## 🔍 Defect Types Detected

The model is trained to detect the following MIG weld defects:

- **🔴 Crack**: Linear discontinuities that can compromise structural integrity
- **🟡 Excess Reinforcement**: Excessive weld metal above the base material surface
- **🔵 Porosity**: Gas pockets trapped within the weld metal
- **🟠 Spatter**: Metal particles expelled during welding
- **🟢 Welding Seam**: Proper weld identification for quality verification

## 📊 Performance Metrics

Our trained model achieves excellent performance on the test dataset:

| Metric | Score |
|--------|-------|
| **Precision** | ~90% |
| **Recall** | ~88% |
| **mAP@0.5** | ~92% |

*Results may vary based on training parameters and dataset characteristics*

## 🏗️ Architecture

- **Base Model**: YOLOv8 (You Only Look Once v8)
- **Variants**: Support for `yolov8n.pt` (nano) and `yolov8s.pt` (small)
- **Training Method**: Transfer learning from pretrained COCO weights
- **Input Resolution**: 640x640 pixels
- **Framework**: Ultralytics YOLOv8

## 📁 Dataset Structure

The dataset is organized in YOLO format with the following structure:

```
dataset/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
└── labels/
    ├── train/          # Training annotations (.txt files)
    ├── val/            # Validation annotations (.txt files)
    └── test/           # Test annotations (.txt files)
```

### Data Configuration

The `data.yaml` file contains:
- Paths to training, validation, and test datasets
- Class names and their corresponding indices
- Number of classes (5 defect types)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Sufficient disk space for dataset and model weights

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/debraj-m/MIGWeld_Defect_Detection.git
cd MIGWeld_Defect_Detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
yolo --version
```

### Training

Train the model on your dataset:

```bash
# Basic training with YOLOv8 nano
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640

# Advanced training with custom parameters
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640 batch=16 lr0=0.01
```

### Validation

Evaluate the trained model:

```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
```

### Inference

Run predictions on new images:

```bash
# Single image prediction
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg

# Batch prediction
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=path/to/images/

# Real-time webcam detection
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=0
```

## 📝 Usage Examples

### Python API

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run inference
results = model('path/to/weld_image.jpg')

# Process results
for result in results:
    # Get bounding boxes, scores, and class predictions
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    
    # Print detections
    for box, score, cls in zip(boxes, scores, classes):
        class_name = model.names[int(cls)]
        print(f"Detected {class_name} with confidence {score:.2f}")
```

### Custom Training Script

```python
from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='migweld_detection'
)

# Validate
model.val()

# Export for deployment
model.export(format='onnx')
```

## 📈 Training Configuration

### Recommended Hyperparameters

```yaml
# Training parameters
epochs: 50-100
batch_size: 16
learning_rate: 0.01
image_size: 640

# Data augmentation
flipud: 0.5
fliplr: 0.5
mosaic: 1.0
mixup: 0.1
```

### Hardware Requirements

- **Minimum**: 8GB RAM, GTX 1060 or equivalent
- **Recommended**: 16GB+ RAM, RTX 3070 or better
- **Training time**: ~2-4 hours on RTX 3070 (50 epochs)

## 🔧 Model Optimization

### For Production Deployment

```bash
# Export to ONNX for cross-platform deployment
yolo export model=best.pt format=onnx

# Export to TensorRT for NVIDIA GPUs
yolo export model=best.pt format=engine

# Export to CoreML for Apple devices
yolo export model=best.pt format=coreml
```

### Model Quantization

```python
# INT8 quantization for edge deployment
model = YOLO('best.pt')
model.export(format='onnx', int8=True)
```

## 📊 Results and Visualization

### Training Metrics

The training process generates comprehensive metrics including:
- Loss curves (box, object, classification)
- Precision-Recall curves
- Confusion matrix
- F1-score curves

### Sample Detections

Results are saved with bounding boxes and confidence scores:
- High-confidence detections (>0.5) are displayed in green
- Medium-confidence detections (0.3-0.5) in yellow
- All detections include class labels and confidence percentages

## 🚀 Future Enhancements

### Planned Features
- [ ] **Edge Deployment**: Optimization for Raspberry Pi and NVIDIA Jetson
- [ ] **Real-time Processing**: Live camera feed integration
- [ ] **Model Compression**: Pruning and quantization for faster inference
- [ ] **Dataset Expansion**: Additional defect types and welding processes
- [ ] **Active Learning**: Automated annotation pipeline
- [ ] **Web Interface**: Browser-based detection tool
- [ ] **Mobile App**: Smartphone-based inspection tool

### Research Directions
- Multi-modal fusion with thermal imaging
- 3D defect reconstruction
- Severity assessment and grading
- Temporal analysis for process monitoring

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [Welding Defect Classification Standards](https://www.aws.org/)

## 🙏 Acknowledgments

This project was developed as part of an AI/ML internship program. Special thanks to:
- Mentors and supervisors for guidance and support
- The welding industry experts for domain knowledge
- The open-source community for tools and frameworks
- Ultralytics team for the excellent YOLOv8 implementation

## 📞 Contact

- **Author**: [Debraj M](https://github.com/debraj-m)
- **Project Link**: [https://github.com/debraj-m/MIGWeld_Defect_Detection](https://github.com/debraj-m/MIGWeld_Defect_Detection)
- **Issues**: [GitHub Issues](https://github.com/debraj-m/MIGWeld_Defect_Detection/issues)

---

**⭐ If this project helped you, please give it a star!**
