from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data="/content/6d-pose-estimation/data/dataset_yolo/data.yaml",
    imgsz=640,
    epochs=10,
    batch=16,
    freeze=10,
    workers=4,
    project="/content/drive/MyDrive/machine_learning_project",
    name="yolo_8020"
)
print(results)
print(results.save_dir)