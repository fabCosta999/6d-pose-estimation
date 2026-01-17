import torch
from torch import optim
from tqdm import tqdm
from src.models.resnet import PoseResNet
from src.models.losses.geodesic_loss import  SymmetryAwareGeodesicLoss
from src.utils.quaternions import rotation_error_deg_symmetry_aware
from src.datasets.scene import LinemodSceneDataset, GTDetections
from src.datasets.resnet import ResNetDataset
import random 
from torch.utils.data import DataLoader, Subset
import csv
import os
import torchvision.utils as vutils


print("[INFO] starting...")

dataset_root = "/content/6d-pose-estimation/data/Linemod_preprocessed"
batch_size = 64
num_epochs = 50
lr = 1e-4
log_dir = "/content/drive/MyDrive/machine_learning_project/logs"
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, "training_log.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "train_angle_deg",
        "val_angle_deg",
        "lr"
    ])
img_log_dir = os.path.join(log_dir, "sample_inputs")
os.makedirs(img_log_dir, exist_ok=True)
print("[INFO] constructing datasets...")
scene_ds = LinemodSceneDataset(
        dataset_root=dataset_root,
        split="train"
    )
gt_detections = GTDetections(scene_ds)
train_ds = ResNetDataset(
        scene_dataset=scene_ds,
        detection_provider=gt_detections,
        img_size=224,
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
model = PoseResNet(pretrained=True).to(device)
criterion = SymmetryAwareGeodesicLoss(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=15, gamma=0.5
)

best_loss = float("inf")

for epoch in range(num_epochs):

    # =========================
    # TRAIN
    # =========================
    model.train()
    running_train_loss = 0.0
    running_angle = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")

    for i, batch in enumerate(pbar):
        rgb = batch["rgb"].to(device)       
        q_gt = batch["rotation"].to(device)  
        label = batch["label"].to(device)

        optimizer.zero_grad()

        q_pred = model(rgb)
        loss = criterion(q_pred, q_gt, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running_train_loss += loss.item() * rgb.size(0)
        pbar.set_postfix(train_loss=loss.item())
        with torch.no_grad():
            angle = rotation_error_deg_symmetry_aware(q_pred, q_gt, label, device)
            running_angle += angle.sum().item()

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
    train_angle = running_angle / len(train_loader.dataset)

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    running_valid_loss = 0.0
    running_valid_angle = 0.0

    with torch.no_grad():
        pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]")

        for batch in pbar:
            rgb = batch["rgb"].to(device)
            q_gt = batch["rotation"].to(device)
            label = batch["label"].to(device)

            q_pred = model(rgb)
            loss = criterion(q_pred, q_gt, label)

            running_valid_loss += loss.item() * rgb.size(0)
            pbar.set_postfix(val_loss=loss.item())
            angle = rotation_error_deg_symmetry_aware(q_pred, q_gt, label, device)
            running_valid_angle += angle.sum().item()

    valid_epoch_loss = running_valid_loss / len(valid_loader.dataset)
    valid_angle = running_valid_angle / len(valid_loader.dataset)
    
    scheduler.step()

    # =========================
    # LOGGING + CHECKPOINT
    # =========================
    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {train_epoch_loss:.6f} | "
        f"Val Loss: {valid_epoch_loss:.6f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e} |"
        f"angle error: {valid_angle}"
    )

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            train_epoch_loss,
            valid_epoch_loss,
            train_angle,
            valid_angle,
            scheduler.get_last_lr()[0]
        ])

    if valid_epoch_loss < best_loss:
        best_loss = valid_epoch_loss
        torch.save(model.state_dict(), "/content/drive/MyDrive/machine_learning_project/pose_resnet_best.pth")
        print("Saved new best model")
    torch.save(model.state_dict(), "/content/drive/MyDrive/machine_learning_project/pose_resnet_last.pth")

