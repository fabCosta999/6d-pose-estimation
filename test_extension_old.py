import trimesh
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import numpy as np
from src.datasets.scene import LinemodSceneDataset
from src.models.rgbd_posenet import RGBDFusionNet
from src.models.rgbd_encoder_decoder import EncoderDecoderWeightsNet
import torchvision.transforms as T
from src.utils.linemod_symmetries import LINEMOD_SYMMETRIES, SYMMETRIC_QUATS, SymmetryType

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


def quat_mul(q1, q2):
    # q = q1 ⊗ q2  (wxyz)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], device=q1.device)



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


def depth_to_points(depth, K, uv_grid):
    """
    depth:   [B, 1, H, W]
    uv_grid: [B, H, W, 2]
    K:       [3, 3]
    return:  [B, H, W, 3]
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = uv_grid[..., 0]
    v = uv_grid[..., 1]
    z = depth.squeeze(1)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return torch.stack([x, y, z], dim=-1)


def build_uv_grid(box, H, W, device):
    """
    box: [B, 4] -> (x, y, w, h) in pixel immagine
    return: uv_grid [B, H, W, 2]
    """
    B = box.shape[0]

    x, y, bw, bh = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

    i = torch.arange(H, device=device).float()
    j = torch.arange(W, device=device).float()
    ii, jj = torch.meshgrid(i, j, indexing="ij")

    ii = ii.unsqueeze(0).expand(B, -1, -1)
    jj = jj.unsqueeze(0).expand(B, -1, -1)

    u = x[:, None, None] + (jj + 0.5) * bw[:, None, None] / W
    v = y[:, None, None] + (ii + 0.5) * bh[:, None, None] / H

    return torch.stack([u, v], dim=-1)  # [B, H, W, 2]


def weighted_translation(points_3d, weights):
    """
    points_3d: [B, H, W, 3]
    weights:   [B, 1, H, W]
    """
    weights = weights.permute(0, 2, 3, 1)  # [B, H, W, 1]
    t = (points_3d * weights).sum(dim=(1,2))
    return t


def spatial_softmax(weight_map, mask=None, tau=0.05):
    B, _, H, W = weight_map.shape
    w = weight_map.view(B, -1) / tau

    if mask is not None:
        m = mask.view(B, -1)
        w = w.masked_fill(m == 0, -1e9)

    w = torch.softmax(w, dim=1)
    return w.view(B, 1, H, W)



def entropy_loss_all(weights, eps=1e-8):
    B = weights.shape[0]
    w = weights.view(B, -1)
    entropy = -(w * torch.log(w + eps)).sum(dim=1)
    return entropy.mean()

def make_coord_grid(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=0)   # [2, H, W]
    return grid






device = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO
yolo = YOLO("/content/drive/MyDrive/machine_learning_project/yolo11s/detect/train/weights/best.pt")

# ResNet (rotazione)
rot_net = RGBDFusionNet(pretrained=False).to(device)
rot_net.load_state_dict(torch.load("/content/drive/MyDrive/machine_learning_project/rgbd/train/weights/best.pth", map_location=device))
rot_net.eval()

# EncDec (traslazione)
enc_dec_net = EncoderDecoderWeightsNet().to(device)
enc_dec_net.load_state_dict(torch.load("/content/drive/MyDrive/machine_learning_project/enc_dec/train/weights/best.pth", map_location=device))
enc_dec_net.eval()



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


depth_mean = 990.7
depth_std  = 311.8

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
    # LOAD RGB + DEPTH (full)
    # ---------------------------
    img = Image.open(scene["img_path"]).convert("RGB")
    depth_img = Image.open(scene["depth_path"])

    # ---------------------------
    # CROP
    # ---------------------------
    crop_rgb_img = crop_rgb(img, gt_bbox)
    if crop_rgb_img is None:
        continue

    x, y, w, h = gt_bbox
    x1 = int(max(0, x))
    y1 = int(max(0, y))
    x2 = int(min(depth_img.width,  x + w))
    y2 = int(min(depth_img.height, y + h))
    if x2 <= x1 or y2 <= y1:
        continue

    crop_depth_img = depth_img.crop((x1, y1, x2, y2))

    # =========================================================
    # ROTATION — RGBD Fusion Net (224)
    # =========================================================
    rgb_224 = resnet_tf(crop_rgb_img).unsqueeze(0).to(device)

    depth_224 = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])(crop_depth_img).unsqueeze(0).to(device)

    depth_224 = depth_224 * ds.depth_scale
    depth_224 = (depth_224 - depth_mean) / depth_std

    with torch.no_grad():
        q_pred = rot_net(rgb_224, depth_224)[0]

    R_pred = quat_to_rot(q_pred)

    # =========================================================
    # TRANSLATION — Encoder Decoder (64)
    # =========================================================
    rgb_64 = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])(crop_rgb_img).unsqueeze(0).to(device)

    depth_64 = T.Compose([
        T.Resize((64, 64), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])(crop_depth_img).unsqueeze(0).to(device)

    depth_64 = depth_64 * ds.depth_scale
    depth_64 = (depth_64 - depth_mean) / depth_std

    coord = make_coord_grid(64, 64, device).unsqueeze(0)
    enc_in = torch.cat([rgb_64, depth_64, coord], dim=1)

    with torch.no_grad():
        logits = enc_dec_net(enc_in)

    # denormalize depth
    un_depth = depth_64 * depth_std + depth_mean
    valid_mask = (un_depth > 10).float()

    weights = spatial_softmax(logits, valid_mask)

    K = scene["cam_intrinsics"].to(device)
    box = torch.tensor(gt_bbox, device=device).unsqueeze(0)
    uv_grid = build_uv_grid(box, 64, 64, device)

    points_3d = depth_to_points(un_depth, K, uv_grid)
    t_pred = weighted_translation(points_3d, weights)[0]

    # =========================================================
    # ADD-S (con simmetrie)
    # =========================================================
    pts = models_3d[obj_id]

    if LINEMOD_SYMMETRIES.get(obj_id, SymmetryType.NONE) == SymmetryType.DISCRETE:
        errs = []
        for q_sym in SYMMETRIC_QUATS[obj_id]:
            q_gt_sym = quat_mul(q_gt, q_sym.to(device))
            R_gt_sym = quat_to_rot(q_gt_sym)

            errs.append(add_s(
                pts,
                R_pred, t_pred,
                R_gt_sym, t_gt,
            ))

        err = torch.stack(errs).min()
    else:
        R_gt = quat_to_rot(q_gt)
        err = add_s(
            pts,
            R_pred, t_pred,
            R_gt, t_gt,
        )

    errors.append(err.item())



errors = torch.tensor(errors)
print(f"ADD-S mean: {errors.mean():.2f} mm")
print(f"ADD-S median: {errors.median():.2f} mm")