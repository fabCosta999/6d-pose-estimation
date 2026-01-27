import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np
from src.datasets.scene import LinemodSceneDataset
from src.models.rgbd_posenet import RGBDFusionNet
from src.models.rgbd_encoder_decoder import EncoderDecoderWeightsNet
import torchvision.transforms as T
from src.utils.linemod_symmetries import LINEMOD_SYMMETRIES, SYMMETRIC_QUATS, SymmetryType
from src.utils.models3d import load_linemod_models, add_metric
from src.utils.quaternions import quaternion_to_rotation_matrix, quat_mul
from src.utils.grid import make_coord_grid, spatial_softmax, build_uv_grid
from src.utils.pinhole import depth_to_points, weighted_translation
from collections import defaultdict
import os
import csv

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

def bbox_invalid(bbox):
    x, y, w, h = bbox
    return (w <= 1) or (h <= 1)


log = defaultdict(lambda: {
    "adds": [],
    "bbox_missing": 0,
    "false_positive": 0,
    "bbox_invalid": 0,
    "depth_missing": 0,
    "total": 0,
})



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
    stream=False,
    save=False,
)


depth_mean = 990.7
depth_std  = 311.8

for r, scene in zip(results, ds):

    # ---------------------------
    # GT
    # ---------------------------
    q_gt = scene["rotation"].to(device)
    t_gt = scene["translation"].to(device)
    obj_class = scene["label"]
    obj_id = LinemodSceneDataset.CLASSES[obj_class]
    log[obj_id]["total"] += 1

    # ---------------------------
    # YOLO bbox
    # ---------------------------

    boxes = r.boxes

    if boxes is None or len(boxes) == 0:
        log[obj_id]["bbox_missing"] += 1
        continue

    xyxy = boxes.xyxy        
    cls  = boxes.cls.long() 
    conf = boxes.conf

    mask = cls == obj_class
    if mask.sum() == 0:
        log[obj_id]["bbox_missing"] += 1
        log[obj_id]["false_positive"] += len(cls)
        continue


    log[obj_id]["false_positive"] += int((~mask).sum())
    if mask.sum() > 1:
        log[obj_id]["false_positive"] += int(mask.sum() - 1)

    idxs = torch.where(mask)[0]
    best = idxs[conf[idxs].argmax()]

    x1, y1, x2, y2 = xyxy[best]

    bbox = (
        x1.item(),
        y1.item(),
        (x2 - x1).item(),
        (y2 - y1).item(),
    )

    if bbox_invalid(bbox):
        log[obj_id]["bbox_invalid"] += 1
        continue




    # ---------------------------
    # LOAD RGB + DEPTH (full)
    # ---------------------------
    img = Image.open(scene["img_path"]).convert("RGB")
    depth_img = Image.open(scene["depth_path"])

    

    # ---------------------------
    # CROP
    # ---------------------------
    crop_rgb_img = crop_rgb(img, bbox)
    if crop_rgb_img is None:
        log[obj_id]["bbox_invalid"] += 1
        continue

    x, y, w, h = bbox
    x1 = int(max(0, x))
    y1 = int(max(0, y))
    x2 = int(min(depth_img.width,  x + w))
    y2 = int(min(depth_img.height, y + h))
    if x2 <= x1 or y2 <= y1:
        log[obj_id]["bbox_invalid"] += 1
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

    R_pred = quaternion_to_rotation_matrix(q_pred)

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

    un_depth = depth_64 * depth_std + depth_mean
    valid_mask = (un_depth > 10).float()

    if valid_mask.sum() == 0:
        log[obj_id]["depth_missing"] += 1
        continue

    weights = spatial_softmax(logits, valid_mask)

    K = scene["cam_intrinsics"].to(device)
    box = torch.tensor(bbox, device=device).unsqueeze(0)
    uv_grid = build_uv_grid(box, 64, 64, device)

    points_3d = depth_to_points(un_depth, K, uv_grid)
    t_pred = weighted_translation(points_3d, weights)[0]

    # =========================================================
    # ADD-S 
    # =========================================================
    pts = models_3d[obj_id]

    if LINEMOD_SYMMETRIES.get(obj_class, SymmetryType.NONE) == SymmetryType.DISCRETE:
        errs = []
        for q_sym in SYMMETRIC_QUATS[obj_class]:
            q_gt_sym = quat_mul(q_gt, q_sym.to(device))
            R_gt_sym = quaternion_to_rotation_matrix(q_gt_sym)

            errs.append(add_metric(
                pts,
                R_pred, t_pred,
                R_gt_sym, t_gt,
            ))

        err = torch.stack(errs).min()
    else:
        R_gt = quaternion_to_rotation_matrix(q_gt)
        err = add_metric(
            pts,
            R_pred, t_pred,
            R_gt, t_gt,
        )

    log[obj_id]["adds"].append(err.item())
    errors.append(err.item())



errors = torch.tensor(errors)
print(f"ADD-S mean: {errors.mean():.2f} mm")
print(f"ADD-S median: {errors.median():.2f} mm")

out_dir = "/content/drive/MyDrive/machine_learning_project/extension_results"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "linemod_eval_extension.csv")
adds_txt_path = os.path.join(out_dir, "linemod_adds.txt")

with open(csv_path, "w", newline="") as f_csv, \
     open(adds_txt_path, "w") as f_txt:

    writer = csv.writer(f_csv)

    writer.writerow([
        "obj_id",
        "num_samples",
        "adds_mean_mm",
        "adds_median_mm",
        "bbox_missing",
        "false_positive",
        "bbox_invalid",
        "depth_missing",
    ])

    for obj_id, d in sorted(log.items()):
        adds = np.array(d["adds"])

        # ---- CSV summary ----
        writer.writerow([
            obj_id,
            d["total"],
            adds.mean() if len(adds) > 0 else np.nan,
            np.median(adds) if len(adds) > 0 else np.nan,
            d["bbox_missing"],
            d["false_positive"],
            d["bbox_invalid"],
            d["depth_missing"],
        ])

        # ---- TXT: all errors ----
        for e in adds:
            f_txt.write(f"{obj_id} {e:.6f}\n")