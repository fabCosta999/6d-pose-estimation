import torch
from torch import optim
from tqdm import tqdm
from src.models.rgbd_encoder_decoder import EncoderDecoderWeightsNet
from src.datasets.scene import LinemodSceneDataset, GTDetections
from src.datasets.rgbd import RGBDDataset
import random 
from torch.utils.data import DataLoader, Subset
import csv
import os
import torchvision.utils as vutils


def depth_to_points(depth, K, uv_grid):
    """
    depth:   [B, 1, H, W]
    uv_grid: [B, H, W, 2]
    K:       [3, 3]
    return:  [B, H, W, 3]
    """
    fx = K[0, 0].item()
    fy = K[1, 1].item()
    cx = K[0, 2].item()
    cy = K[1, 2].item()

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


def spatial_softmax(weight_map, mask=None):
    B, _, H, W = weight_map.shape
    w = weight_map.view(B, -1)

    if mask is not None:
        m = mask.view(B, -1)
        w = w.masked_fill(m == 0, -1e9)

    w = torch.softmax(w, dim=1)
    return w.view(B, 1, H, W)



print("[INFO] starting...")

dataset_root = "/content/6d-pose-estimation/data/Linemod_preprocessed"
batch_size = 64
num_epochs = 30
lr = 1e-4
log_dir = "/content/drive/MyDrive/machine_learning_project/enc_dec_logs"
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, "training_log.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "train_error",
        "val_error",
        "lr"
    ])
img_log_dir = os.path.join(log_dir, "sample_inputs")
os.makedirs(img_log_dir, exist_ok=True)
print("[INFO] constructing datasets...")
scene_ds = LinemodSceneDataset(
        dataset_root=dataset_root,
        split="train"
    )
cam_intrinsics = scene_ds[0]["cam_intrinsics"]
print(cam_intrinsics)
"""gt_detections = GTDetections(scene_ds)
train_ds = RGBDDataset(
        scene_dataset=scene_ds,
        detection_provider=gt_detections,
        img_size=64,
        padding=0
    )
print("[INFO] datasets ready")
indices = list(range(len(train_ds)))
random.seed(42)
random.shuffle(indices)
split = int(0.9 * len(indices))
train_idx = indices[:split]
valid_idx = indices[split:]
train_subset = Subset(train_ds, train_idx)
valid_subset = Subset(train_ds, valid_idx)
train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
valid_loader = DataLoader(
    valid_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoderWeightsNet().to(device)
#criterion = # MSE?
criterion = torch.nn.SmoothL1Loss(beta=0.01)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5
)

best_loss = float("inf")

for epoch in range(num_epochs):

    # =========================
    # TRAIN
    # =========================
    model.train()
    running_train_loss = 0.0
    running_error = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")

    for i, batch in enumerate(pbar):
        rgb = batch["rgb"].to(device)        # [B, 3, 64, 64]
        depth = batch["depth"].to(device)    # [B, 1, 64, 64]
        box = torch.stack(batch["bbox"]).to(device)   # [B, 4]
        t_gt = batch["translation"].to(device)  # [B, 3]
        
        optimizer.zero_grad()

        weight_map = model(torch.cat([rgb, depth], dim=1))           # [B,1,64,64]
        valid_mask = (depth > 0).float()

        weights = spatial_softmax(weight_map, valid_mask)

        B, _, H, W = depth.shape
        uv_grid = build_uv_grid(box, H, W, device)

        points_3d = depth_to_points(
            depth,
            cam_intrinsics.to(device),
            uv_grid
        )

        t_pred = weighted_translation(points_3d, weights)
        loss = criterion(t_pred, t_gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running_train_loss += loss.item() * rgb.size(0)
        pbar.set_postfix(train_loss=loss.item())
        with torch.no_grad():
            error = torch.norm(t_pred - t_gt, dim=1) 
            running_error += error.sum().item()

        if epoch == 0:
            save_n = min(8, rgb.size(0))  
            grid = vutils.make_grid(
                rgb[:save_n].cpu(),
                nrow=4,
                normalize=True,
                scale_each=True
            )
            vutils.save_image(
                grid,
                os.path.join(img_log_dir, f"train_crops_epoch0_batch{i:04d}.png")
            )

    

    train_epoch_loss = running_train_loss / len(train_loader.dataset)
    train_error = running_error / len(train_loader.dataset)

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    running_valid_loss = 0.0
    running_valid_error = 0.0

    with torch.no_grad():
        pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]")

        for batch in pbar:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            box = torch.stack(batch["bbox"]).to(device)   # [B, 4]
            t_gt = batch["translation"].to(device)

            weight_map = model(torch.cat([rgb, depth], dim=1))
            valid_mask = (depth > 0).float()
            weights = spatial_softmax(weight_map, valid_mask)
            B, _, H, W = depth.shape
            uv_grid = build_uv_grid(box, H, W, device)
            points_3d = depth_to_points(
            depth,
            cam_intrinsics.to(device),
            uv_grid
            )

            t_pred = weighted_translation(points_3d, weights)
            loss = criterion(t_pred, t_gt)

            running_valid_loss += loss.item() * rgb.size(0)
            pbar.set_postfix(val_loss=loss.item())
            error = torch.norm(t_pred - t_gt, dim=1)
            running_valid_error += error.sum().item()

    valid_epoch_loss = running_valid_loss / len(valid_loader.dataset)
    valid_error = running_valid_error / len(valid_loader.dataset)
    
    scheduler.step()

    # =========================
    # LOGGING + CHECKPOINT
    # =========================
    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {train_epoch_loss:.6f} | "
        f"Val Loss: {valid_epoch_loss:.6f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e} |"
        f"error: {valid_error}"
    )

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            train_epoch_loss,
            valid_epoch_loss,
            train_error,
            valid_error,
            scheduler.get_last_lr()[0]
        ])

    if valid_epoch_loss < best_loss:
        best_loss = valid_epoch_loss
        torch.save(model.state_dict(), "/content/drive/MyDrive/machine_learning_project/enc_dec_best.pth")
        print("Saved new best model")
    torch.save(model.state_dict(), "/content/drive/MyDrive/machine_learning_project/enc_dec_last.pth")

"""