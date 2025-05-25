from ultralytics import YOLO

model = YOLO(r"C:\Users\debra\Desktop\Dataset\yolov8n.pt")

model.train(
    data=r"C:\Users\debra\Desktop\Dataset\Drive link Dataset\driveLink.yaml",
    epochs=10,
    imgsz=640,
    batch=8
)
