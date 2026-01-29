import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.datasets.resnet import ResNetDataset
from src.datasets.scene import LinemodSceneDataset, GTDetections
from src.models.resnet import RotationNet
from src.utils.save_results import show_rotation_results
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from src.utils.quaternions import rotation_error_deg_symmetry_aware
import os
import csv
from torchvision.utils import save_image


def main(args):
    results_dir = args.out_dir

    best_per_class = {}
    worst_per_class = {}

    scene_ds = LinemodSceneDataset(args.data_root, split="test")
    dp = GTDetections(scene_ds)
    test_ds = ResNetDataset(
        scene_dataset=scene_ds,
        detection_provider=dp,
        img_size=224,
        padding=0
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RotationNet(pretrained=False)
    model = model.to(device)
    weight_path = args.resnet_weights

    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(" Model weights loaded.")
    except:
        print(" Loading with strict=False...")
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)


    model.eval()

    errors_per_class = defaultdict(list)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")

        for batch in pbar:
            rgb = batch["rgb"].to(device)
            q_gt = batch["rotation"].to(device)
            labels = batch["label"]  # CPU ok

            q_pred = model(rgb)

            errors = rotation_error_deg_symmetry_aware(q_pred, q_gt, labels, device)

            
            for i in range(len(errors)):
                err = float(errors[i].cpu().item())
                cls = int(labels[i].item())

                errors_per_class[cls].append(err)

                # ---------- BEST ----------
                if cls not in best_per_class or err < best_per_class[cls]["error"]:
                    best_per_class[cls] = {
                        "error": err,
                        "rgb": rgb[i].cpu(),
                    }

                # ---------- WORST ----------
                if cls not in worst_per_class or err > worst_per_class[cls]["error"]:
                    worst_per_class[cls] = {
                        "error": err,
                        "rgb": rgb[i].cpu(),
                    }

    show_rotation_results(errors_per_class, results_dir, best_per_class, worst_per_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet_weights", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/Linemod_preprocessed")
    parser.add_argument("--out_dir", type=str, default="test_resnet")
    args = parser.parse_args()
    main(args)
