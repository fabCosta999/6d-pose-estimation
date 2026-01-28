import argparse
import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np
from src.datasets.scene import LinemodSceneDataset
from src.models.rgbd_posenet import DepthRotationNet
from src.models.rgbd_translation import DepthTranslationNet
import torchvision.transforms as T
from src.utils.linemod_symmetries import LINEMOD_SYMMETRIES, SYMMETRIC_QUATS, SymmetryType
from src.utils.models3d import load_linemod_models, add_metric
from src.utils.quaternions import quaternion_to_rotation_matrix, quat_mul
from src.utils.grid import make_coord_grid
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
        "false_positive": 0,
        "bbox_invalid": 0,
        "depth_missing": 0,
        "total": 0,
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = LinemodSceneDataset(args.data_root, split="test")
    models_3d = load_linemod_models(
        f"{args.data_root}/models",
        device=device
    )

    depth_mean = 990.7
    depth_std  = 311.8

    # YOLO
    yolo = YOLO(args.yolo_model)

    # RGBD (Rotation)
    rot_net = DepthRotationNet(pretrained=False).to(device)
    rot_net.load_state_dict(torch.load(args.rgbd_pose_model, map_location=device))
    rot_net.eval()

    # EncDec (translation)
    enc_dec_net = DepthTranslationNet(depth_mean, depth_std).to(device)
    enc_dec_net.end_dec.load_state_dict(torch.load(args.enc_dec_model, map_location=device))
    enc_dec_net.eval()

    errors = []

    results = yolo.predict(
        source=f"{args.yolo_dataset}/test/images",
        imgsz=640,
        batch=16,
        device=device,
        stream=False,
        save=False,
    )

    for r, scene in zip(results, ds):

        # =========================================================
        # GT
        # =========================================================
        q_gt = scene["rotation"].to(device)
        t_gt = scene["translation"].to(device)
        obj_class = scene["label"]
        obj_id = LinemodSceneDataset.CLASSES[obj_class]
        log[obj_id]["total"] += 1

        # =========================================================
        # YOLO bbox
        # =========================================================
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

        # =========================================================
        # LOAD RGB + DEPTH
        # =========================================================
        img = Image.open(scene["img_path"]).convert("RGB")
        depth_img = Image.open(scene["depth_path"])

    
        # =========================================================
        # CROP
        # =========================================================
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

        depth_64 = (depth_64 - depth_mean) / depth_std

        coord = make_coord_grid(64, 64, device).unsqueeze(0)

        box = torch.tensor(bbox, device=device).unsqueeze(0)
        
        K = scene["cam_intrinsics"].to(device)

        with torch.no_grad():
            _, t_pred = enc_dec_net(rgb_64, depth_64, coord, box, K)
            t_pred = t_pred[0]

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
    show_pipeline_results(errors, results_dir, log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_model", type=str, required=True)
    parser.add_argument("--rgbd_pose_model", type=str, required=True)
    parser.add_argument("--enc_dec_model", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/Linemod_preprocessed")
    parser.add_argument("--yolo_dataset", type=str, default="data/dataset_yolo")
    parser.add_argument("--out_dir", type=str, default="test_extension")
    args = parser.parse_args()
    main(args)