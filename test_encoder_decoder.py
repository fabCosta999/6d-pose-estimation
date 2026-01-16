import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.datasets.rgbd import RGBDDataset
from src.datasets.scene import LinemodSceneDataset, GTDetections
from src.models.rgbd_encoder_decoder import EncoderDecoderWeightsNet
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from src.utils.quaternions import rotation_error_deg_symmetry_aware
import os
import csv
from torchvision.utils import save_image


def overlay_pixel_map(rgb, pixel_map, alpha=0.6):
    """
    rgb: Tensor [3, H, W] in [0,1]
    pixel_map: Tensor [H, W] in [0,1]
    """
    rgb = rgb.clone()

    if pixel_map.dim() == 3:
        pixel_map = pixel_map.squeeze(0)

    pixel_map = pixel_map.clamp(0, 1)

    # colormap semplice: rosso
    heat = torch.zeros_like(rgb)
    heat[0] = pixel_map  # red channel

    overlay = (1 - alpha) * rgb + alpha * heat
    return overlay.clamp(0, 1)


def make_coord_grid(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=0)   # [2, H, W]
    return grid

def spatial_softmax(weight_map, mask=None, tau=0.05):
    B, _, H, W = weight_map.shape
    w = weight_map.view(B, -1) / tau

    if mask is not None:
        m = mask.view(B, -1)
        w = w.masked_fill(m == 0, -1e9)

    w = torch.softmax(w, dim=1)
    return w.view(B, 1, H, W)


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


def weighted_translation(points_3d, weights):
    """
    points_3d: [B, H, W, 3]
    weights:   [B, 1, H, W]
    """
    weights = weights.permute(0, 2, 3, 1)  # [B, H, W, 1]
    t = (points_3d * weights).sum(dim=(1,2))
    return t



results_dir = "/content/drive/MyDrive/machine_learning_project/eval_results_enc_dec"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f"{results_dir}/best", exist_ok=True)
os.makedirs(f"{results_dir}/worst", exist_ok=True)

best_per_class = {}
worst_per_class = {}


scene_ds = LinemodSceneDataset("data/Linemod_preprocessed", split="test")
dp = GTDetections(scene_ds)
test_ds = RGBDDataset(
    scene_dataset=scene_ds,
    detection_provider=dp,
    img_size=64,
    padding=0
)
cam_intrinsics = test_ds[0]["cam_intrinsics"]
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoderWeightsNet()
model = model.to(device)
weight_path = "/content/drive/MyDrive/machine_learning_project/enc_dec/enc_dec_best.pth"

try:
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(" Model weights loaded.")
except:
    print(" Loading with strict=False...")
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)


model.eval()

errors_per_class = defaultdict(list)
errors_xyz_per_class = defaultdict(lambda: {"x": [], "y": [], "z": [], "l2": []})
coord_grid = make_coord_grid(64, 64, device)  # [2,64,64]


with torch.no_grad():
    pbar = tqdm(test_loader, desc="Testing")

    for batch in pbar:
        rgb = batch["rgb"].to(device)        # [B, 3, 64, 64]
        depth = batch["depth"].to(device)    # [B, 1, 64, 64]
        box = torch.stack(batch["bbox"], dim=1).to(device)   # [B, 4]
        t_gt = batch["translation"].to(device)  # [B, 3]
        B = rgb.shape[0]
        coord = coord_grid.unsqueeze(0).repeat(B, 1, 1, 1)
        logits = model(torch.cat([rgb, depth, coord], dim=1))   
        un_normalized_depth = depth * test_ds.depth_std + test_ds.depth_mean        
        valid_mask = (un_normalized_depth > 10).float() # 10 mm

        weights = spatial_softmax(logits, valid_mask)
        

        B, _, H, W = depth.shape
        uv_grid = build_uv_grid(box, H, W, device)

        points_3d = depth_to_points(
            un_normalized_depth,
            cam_intrinsics.to(device),
            uv_grid
        )

        t_pred = weighted_translation(points_3d, weights)
        # --- errori ---
        t_err = t_pred - t_gt                    # [B, 3]
        err_x = t_err[:, 0].abs()
        err_y = t_err[:, 1].abs()
        err_z = t_err[:, 2].abs()
        err_l2 = torch.norm(t_err, dim=1)        # errore totale mm

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
            overlay = overlay_pixel_map(rgb_i, w_i)

            # ---------- BEST ----------
            if cls not in best_per_class or el2 < best_per_class[cls]["error"]:
                best_per_class[cls] = {
                    "error": el2,
                    "rgb": rgb_i,
                    "overlay": overlay,
                }

            # ---------- WORST ----------
            if cls not in worst_per_class or el2 > worst_per_class[cls]["error"]:
                worst_per_class[cls] = {
                    "error": el2,
                    "rgb": rgb_i,
                    "overlay": overlay,
                }
print("\n" + "="*70)
print("TRANSLATION ERRORS (mm)")
print("="*70)

all_l2 = []

for cls in sorted(errors_xyz_per_class.keys()):
    ex = np.array(errors_xyz_per_class[cls]["x"])
    ey = np.array(errors_xyz_per_class[cls]["y"])
    ez = np.array(errors_xyz_per_class[cls]["z"])
    el2 = np.array(errors_xyz_per_class[cls]["l2"])

    all_l2.extend(el2.tolist())

    print(f"\nClass {cls}:")
    print(f"  Samples: {len(el2)}")
    print(f"  Mean |x|: {ex.mean():.2f} mm")
    print(f"  Mean |y|: {ey.mean():.2f} mm")
    print(f"  Mean |z|: {ez.mean():.2f} mm")
    print(f"  Mean L2:  {el2.mean():.2f} mm")
    print(f"  Median L2: {np.median(el2):.2f} mm")
    print(f"  Acc < 5mm:  {np.mean(el2 < 5)*100:.2f}%")
    print(f"  Acc < 10mm: {np.mean(el2 < 10)*100:.2f}%")


csv_path = f"{results_dir}/translation_errors_per_class.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "class", "samples",
        "mean_x_mm", "mean_y_mm", "mean_z_mm",
        "mean_l2_mm", "median_l2_mm",
        "acc_<5mm", "acc_<10mm"
    ])

    all_l2 = []

    for cls in sorted(errors_xyz_per_class.keys()):
        ex = np.array(errors_xyz_per_class[cls]["x"])
        ey = np.array(errors_xyz_per_class[cls]["y"])
        ez = np.array(errors_xyz_per_class[cls]["z"])
        el2 = np.array(errors_xyz_per_class[cls]["l2"])
        all_l2.extend(el2.tolist())

        writer.writerow([
            cls, len(el2),
            ex.mean(), ey.mean(), ez.mean(),
            el2.mean(), np.median(el2),
            np.mean(el2 < 5) * 100,
            np.mean(el2 < 10) * 100
        ])

    all_l2 = np.array(all_l2)
    writer.writerow([])
    writer.writerow([
        "OVERALL", len(all_l2),
        "", "", "",
        all_l2.mean(), np.median(all_l2),
        np.mean(all_l2 < 5) * 100,
        np.mean(all_l2 < 10) * 100
    ])


for cls in best_per_class:
    save_image(
        best_per_class[cls]["rgb"],
        f"{results_dir}/best/class_{cls}_err_{best_per_class[cls]['error']:.2f}_rgb.png",
        normalize=True
    )

    save_image(
        best_per_class[cls]["overlay"],
        f"{results_dir}/best/class_{cls}_err_{best_per_class[cls]['error']:.2f}_overlay.png",
        normalize=True
    )

    save_image(
        worst_per_class[cls]["rgb"],
        f"{results_dir}/worst/class_{cls}_err_{worst_per_class[cls]['error']:.2f}_rgb.png",
        normalize=True
    )

    save_image(
        worst_per_class[cls]["overlay"],
        f"{results_dir}/worst/class_{cls}_err_{worst_per_class[cls]['error']:.2f}_overlay.png",
        normalize=True
    )

        
        