from ultralytics import YOLO
from src.datasets.scene import LinemodSceneDataset
import torch

model = YOLO(
    "/content/drive/MyDrive/machine_learning_project/yolo11s_8020/weights/best.pt"
)

ds = LinemodSceneDataset("data/Linemod_preprocessed", split="test")

results = model.predict(
    source="/content/6d-pose-estimation/data/dataset_yolo/test/images",
    imgsz=640,
    batch=16,
    device="cpu",
    save=False
)

for r, scene in zip(results, ds):
    if r.boxes is None or len(r.boxes) == 0:
        continue

    gt = scene["objects"][0]   # per ora 1 oggetto
    bbox = gt["bbox"]
    depth = scene["depth"][0]  # HxW

    K = scene["cam_intrinsics"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x, y, w, h = bbox
    u = x + w / 2
    v = y + h / 2

    u_i = int(round(u))
    v_i = int(round(v))

    # safety clamp (consigliato)
    H, W = depth.shape
    if not (0 <= u_i < W and 0 <= v_i < H):
        continue

    Z = depth[v_i, u_i]

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    t_pred = torch.tensor([X, Y, Z])
    t_gt = gt["translation"]

    print("GT :", t_gt)
    print("PRED:", t_pred)
    break
