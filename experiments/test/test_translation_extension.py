import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.datasets.rgbd import RGBDDataset
from src.datasets.scene import LinemodSceneDataset, GTDetections
from src.models.rgbd_translation import DepthTranslationNet
from src.utils.save_results import show_translation_results
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from src.utils.grid import make_coord_grid

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def enhance_contrast(x, t=0.5, k=12):
    num = sigmoid(k * (x - t)) - sigmoid(-k * t)
    den = sigmoid(k * (1 - t)) - sigmoid(-k * t)
    return num / den

def prepare_weight_map(weight_map):
    if weight_map.dim() == 3:
        weight_map = weight_map.squeeze(0)
    w = weight_map.clone()
    w = w / (w.max() + 1e-8)   
    return enhance_contrast(w.unsqueeze(0))


def main(args):
    results_dir = args.out_dir

    best_per_class = {}
    worst_per_class = {}


    scene_ds = LinemodSceneDataset(args.data_root, split="test")
    dp = GTDetections(scene_ds)
    test_ds = RGBDDataset(
        scene_dataset=scene_ds,
        detection_provider=dp,
        img_size=64,
        padding=0
    )
    cam_intrinsics = scene_ds[0]["cam_intrinsics"]
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthTranslationNet(test_ds.depth_mean, test_ds.depth_std)
    model = model.to(device)
    weight_path = args.enc_dec_model

    try:
        model.end_dec.load_state_dict(torch.load(weight_path, map_location=device))
        print(" Model weights loaded.")
    except:
        print(" Loading with strict=False...")
        model.end_dec.load_state_dict(torch.load(weight_path, map_location=device), strict=False)


    model.eval()

    errors_per_class = defaultdict(list)
    errors_xyz_per_class = defaultdict(lambda: {"x": [], "y": [], "z": [], "l2": []})
    coord_grid = make_coord_grid(64, 64, device)  


    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")

        for batch in pbar:
            rgb = batch["rgb"].to(device)       
            depth = batch["depth"].to(device)    
            box = torch.stack(batch["bbox"], dim=1).to(device)  

            t_gt = batch["translation"].to(device) 

            B = rgb.shape[0]
            coord = coord_grid.unsqueeze(0).repeat(B, 1, 1, 1)

            weights, t_pred = model(rgb, depth, coord, box, cam_intrinsics)   
  
            # --- errors ---
            t_err = t_pred - t_gt                    
            err_x = t_err[:, 0].abs()
            err_y = t_err[:, 1].abs()
            err_z = t_err[:, 2].abs()
            err_l2 = torch.norm(t_err, dim=1)        # total error mm

            for i in range(B):
                cls = int(batch["label"][i].item())

                ex = float(err_x[i].cpu())
                ey = float(err_y[i].cpu())
                ez = float(err_z[i].cpu())
                el2 = float(err_l2[i].cpu())

                errors_xyz_per_class[cls]["x"].append(ex)
                errors_xyz_per_class[cls]["y"].append(ey)
                errors_xyz_per_class[cls]["z"].append(ez)
                errors_xyz_per_class[cls]["l2"].append(el2)

                rgb_i = rgb[i].cpu()
                w_i = weights[i].cpu()
                pixel_map = prepare_weight_map(w_i)

                # ---------- BEST ----------
                if cls not in best_per_class or el2 < best_per_class[cls]["error"]:
                    best_per_class[cls] = {
                        "error": el2,
                        "rgb": rgb_i,
                        "pixel_map": pixel_map,
                    }

                # ---------- WORST ----------
                if cls not in worst_per_class or el2 > worst_per_class[cls]["error"]:
                    worst_per_class[cls] = {
                        "error": el2,
                        "rgb": rgb_i,
                        "pixel_map": pixel_map,
                    }
    
    show_translation_results(errors_xyz_per_class, results_dir, best_per_class, worst_per_class)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc_dec_model", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/Linemod_preprocessed")
    parser.add_argument("--out_dir", type=str, default="test_enc_dec")
    args = parser.parse_args()
    main(args)