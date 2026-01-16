import trimesh
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from src.datasets.scene import LinemodSceneDataset
from src.models.resnet import PoseResNet

import torchvision.transforms as T

resnet_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])


def crop_rgb(img_pil, bbox):
    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return None
    x1 = int(max(0, x))
    y1 = int(max(0, y))
    x2 = int(min(img_pil.width,  x + w))
    y2 = int(min(img_pil.height, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return img_pil.crop((x1, y1, x2, y2))




def load_linemod_models(models_dir, device="cpu"):
    models = {}
    for ply in Path(models_dir).glob("obj_*.ply"):
        obj_id = int(ply.stem.split("_")[1])
        mesh = trimesh.load(ply, process=False)
        pts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        models[obj_id] = pts
    return models



@torch.no_grad()
def add_s(
    model_points,   # (N,3)
    R_pred, t_pred, # (3,3), (3,)
    R_gt,   t_gt,   # (3,3), (3,)
):
    """
    returns scalar ADD-S
    """

    # Transform points
    pts_pred = (R_pred @ model_points.T).T + t_pred
    pts_gt   = (R_gt   @ model_points.T).T + t_gt

    # Pairwise distances (N,N)
    dists = torch.cdist(pts_pred, pts_gt, p=2)

    # closest gt point for each pred point
    min_dists, _ = dists.min(dim=1)

    return min_dists.mean()



def quat_to_rot(q):
    # q: (4,) wxyz
    q = q / q.norm()
    w, x, y, z = q
    return torch.tensor([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], device=q.device)



class PoseEvaluator:
    def __init__(self, models_3d):
        self.models = models_3d  # dict[obj_id] -> (N,3)

    def evaluate(self, obj_id, q_pred, t_pred, q_gt, t_gt):
        R_pred = quat_to_rot(q_pred)
        R_gt   = quat_to_rot(q_gt)

        pts = self.models[obj_id]

        return add_s(
            pts,
            R_pred, t_pred,
            R_gt,   t_gt,
        )




device = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO
yolo = YOLO("/content/drive/MyDrive/machine_learning_project/yolo11s/detect/train/weights/best.pt")

# ResNet (rotazione)
rot_net = PoseResNet(pretrained=False).to(device)
rot_net.load_state_dict(torch.load("/content/drive/MyDrive/machine_learning_project/resnet50/train/weights/best.pth", map_location=device))
rot_net.eval()

# Dataset
ds = LinemodSceneDataset("data/Linemod_preprocessed", split="test")

# Modelli 3D (UNA VOLTA)
models_3d = load_linemod_models(
    "data/Linemod_preprocessed/models",
    device=device
)


errors = []

results = yolo.predict(
    source="/content/6d-pose-estimation/data/dataset_yolo/test/images",
    imgsz=640,
    batch=16,
    device=device,
    stream=True,
    save=False,
)

for r, scene in zip(results, ds):

    # ---------------------------
    # GT
    # ---------------------------
    gt_bbox = scene["bbox"]
    q_gt = scene["rotation"].to(device)
    t_gt = scene["translation"].to(device)
    obj_class = scene["label"]
    obj_id = LinemodSceneDataset.CLASSES[obj_class]

    # ---------------------------
    # TRANSLATION (pinhole, GT bbox)
    # ---------------------------
    depth_img = Image.open(scene["depth_path"])
    depth = torch.tensor(
        depth_img, dtype=torch.float32
    ) * scene["cam_intrinsics"].new_tensor(1.0)

    K = scene["cam_intrinsics"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x, y, w, h = gt_bbox
    u = x + w / 2
    v = y + h / 2

    u_i = int(round(u))
    v_i = int(round(v))

    H, W = depth.shape
    if not (0 <= u_i < W and 0 <= v_i < H):
        continue

    Z = depth[v_i, u_i]
    if Z <= 0:
        continue

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    t_pred = torch.tensor([X, Y, Z], device=device)

    # ---------------------------
    # ROTATION (ResNet)
    # ---------------------------
    img = Image.open(scene["img_path"]).convert("RGB")
    crop = crop_rgb(img, gt_bbox)
    if crop is None:
        continue

    rgb = resnet_tf(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        q_pred = rot_net(rgb)[0]

    # ---------------------------
    # ADD-S
    # ---------------------------
    R_pred = quat_to_rot(q_pred)
    R_gt   = quat_to_rot(q_gt)

    pts = models_3d[obj_id]

    err = add_s(
        pts,
        R_pred, t_pred,
        R_gt,   t_gt,
    )

    errors.append(err.item())
