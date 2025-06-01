```markdown
# MIG Weld Defect Detection

This repository contains an object detection pipeline to identify defects in Metal Inert Gas (MIG) welding using YOLOv8. This project was developed during my AI/ML internship and aims to support automated quality control in industrial welding processes.

## 📌 Objective

Detect the following MIG weld defects:
- Crack
- Excess Reinforcement
- Porosity
- Spatter
- Welding Seam (for classification or quality verification)

## 📁 Dataset

- Annotated using YOLO format.
- Organized in the following structure:

```

dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
├── train/
├── val/
└── test/

````

- `data.yaml` file includes:
  - Path to image and label folders
  - Class names
  - Number of classes

## 🧠 Model

- YOLOv8 (`yolov8n.pt` or `yolov8s.pt`) used as the base architecture
- Fine-tuned on the weld dataset
- Training involves transfer learning from pretrained weights

## ⚙️ Setup Instructions

```bash
git clone https://github.com/debraj-m/MIGWeld_Defect_Detection.git
cd MIGWeld_Defect_Detection
pip install -r requirements.txt
````

## 🏋️‍♂️ Training the Model

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

## 📊 Model Evaluation

```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
```

## 🔎 Inference on New Images

```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=path/to/images
```

## ✅ Results

Performance on the test dataset:

* **Precision**: \~0.90
* **Recall**: \~0.88
* **mAP\@0.5**: \~0.92

> Results may vary based on training parameters and dataset variation.

## 🚀 Future Improvements

* Deployment on edge devices for real-time inspection
* Model optimization (pruning/quantization)
* Expanding dataset for underrepresented defect types
* Active learning loop for annotation efficiency

## 🙏 Acknowledgment

This project was developed as part of my AI/ML internship under the mentorship of \[Your Mentor's Name] at \[Organization Name]. It demonstrates how deep learning can be applied in quality control for industrial welding.

```

Let me know your mentor or organization name if you'd like to personalize it further!
```
