import argparse
import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np
from src.datasets.scene import LinemodSceneDataset
from src.models.resnet import RotationNet
import torchvision.transforms as T
from src.utils.linemod_symmetries import LINEMOD_SYMMETRIES, SYMMETRIC_QUATS, SymmetryType
from src.utils.quaternions import quaternion_to_rotation_matrix, quat_mul
from src.utils.models3d import load_linemod_models, add_metric
from src.utils.save_results import show_pipeline_results
from collections import defaultdict


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


def main(args):
    results_dir = args.out_dir

    log = defaultdict(lambda: {
        "adds": [],
        "bbox_missing": 0,
        "false_positive": 0,    # other class predicted/multiple predictions
        "bbox_invalid": 0,      # w<=0, h<=0, crop None, ecc.
        "depth_missing": 0,     # Z<=0 or out of image
        "total": 0,             # total images per class
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # YOLO
    yolo = YOLO(args.yolo_weights)

    # ResNet 
    rot_net = RotationNet(pretrained=False).to(device)
    rot_net.load_state_dict(torch.load(args.resnet_weights, map_location=device))
    rot_net.eval()

    # Dataset
    ds = LinemodSceneDataset(args.data_root, split="test")

    models_3d = load_linemod_models(
        f"{args.data_root}/models",
        device=device
    )

    errors_adds = []

    results = yolo.predict(
        source=f"{args.yolo_dataset}/test/images",
        imgsz=640,
        batch=16,
        device=device,
        stream=False,
        save=False,
    )

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
        # TRANSLATION (pinhole)
        # ---------------------------
        depth_img = Image.open(scene["depth_path"])
        depth_np = np.array(depth_img).astype(np.float32)
        depth = torch.from_numpy(depth_np)   
        
        K = scene["cam_intrinsics"]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x, y, w, h = bbox
        u = x + w / 2
        v = y + h / 2

        u_i = int(round(u))
        v_i = int(round(v))

        H, W = depth.shape
        if not (0 <= u_i < W and 0 <= v_i < H):
            log[obj_id]["depth_missing"] += 1
            continue

        Z = depth[v_i, u_i]
        if Z <= 0:
            log[obj_id]["depth_missing"] += 1
            continue

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        t_pred = torch.tensor([X, Y, Z], device=device)

        # ---------------------------
        # ROTATION (ResNet)
        # ---------------------------
        img = Image.open(scene["img_path"]).convert("RGB")
        crop = crop_rgb(img, bbox)
        if crop is None:
            log[obj_id]["bbox_invalid"] += 1
            continue

        rgb = resnet_tf(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            q_pred = rot_net(rgb)[0]
        R_pred = quaternion_to_rotation_matrix(q_pred)

        # ---------------------------
        # ADD-S
        # ---------------------------
        pts = models_3d[obj_id]

        if LINEMOD_SYMMETRIES.get(obj_class, SymmetryType.NONE) == SymmetryType.DISCRETE:
            errs = []

            for q_sym in SYMMETRIC_QUATS[obj_class]:
                q_gt_sym = quat_mul(q_gt, q_sym.to(device))
                R_gt_sym = quaternion_to_rotation_matrix(q_gt_sym)

                e = add_metric(
                    pts,
                    R_pred, t_pred,
                    R_gt_sym, t_gt,
                )
                errs.append(e)

            err = torch.stack(errs).min()

        else:
            R_gt = quaternion_to_rotation_matrix(q_gt)
            err = add_metric(
                pts,
                R_pred, t_pred,
                R_gt,   t_gt,
            )

        errors_adds.append(err.item())
        log[obj_id]["adds"].append(err.item())


    errors = torch.tensor(errors_adds)
    show_pipeline_results(errors, results_dir, log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_weights", type=str, required=True)
    parser.add_argument("--resnet_weights", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/Linemod_preprocessed")
    parser.add_argument("--yolo_dataset", type=str, default="data/dataset_yolo")
    parser.add_argument("--out_dir", type=str, default="test_pipeline_baseline")
    args = parser.parse_args()
    main(args)

