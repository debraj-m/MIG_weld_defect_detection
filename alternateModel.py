from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=r"C:\Users\debra\Desktop\Dataset\weld_quality_dataset\data.yaml",
    epochs=10,
    imgsz=640,
    batch=8
)
