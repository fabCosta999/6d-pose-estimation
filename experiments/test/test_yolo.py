from ultralytics import YOLO

model = YOLO(
    "/content/drive/MyDrive/machine_learning_project/yolo11s/detect/train/weights/best.pt"
)

metrics = model.val(
    data="/content/6d-pose-estimation/data/dataset_yolo/data.yaml",
    split="test",
    imgsz=640,
    batch=16
)
