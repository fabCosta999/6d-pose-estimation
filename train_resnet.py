import torch
from torch import optim
from tqdm import tqdm
from src.networks.resnet import PoseResNet, SymmetryAwareGeodesicLoss, rotation_error_deg_symmetry_aware
from src.datasets.scene import LinemodSceneDataset, GTDetections
from src.datasets.resnet import ResNetDataset
import random 
from torch.utils.data import DataLoader, Subset


dataset_root = "/content/6d-pose-estimation/data/Linemod_preprocessed"
batch_size = 32
num_epochs = 50
lr = 1e-4
scene_ds = LinemodSceneDataset(
        dataset_root=dataset_root,
        split="train"
    )
gt_detections = GTDetections(scene_ds)
train_ds = ResNetDataset(
        scene_dataset=scene_ds,
        detection_provider=gt_detections,
        img_size=224,
        padding=0,
        enable_transform=False
    )
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
print("dataloader pronto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PoseResNet(pretrained=True).to(device)
criterion = SymmetryAwareGeodesicLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=15, gamma=0.5
)

num_epochs = 50
best_loss = float("inf")

for epoch in range(num_epochs):
    # =========================
    # TRAIN
    # =========================
    model.train()
    running_train_loss = 0.0
    running_angle = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")

    for batch in pbar:
        rgb = batch["rgb"].to(device)        # [B, 3, 224, 224]
        q_gt = batch["rotation"].to(device)  # [B, 4]
        label = batch["label"].to(device)

        optimizer.zero_grad()

        q_pred = model(rgb)
        loss = criterion(q_pred, q_gt, label)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * rgb.size(0)
        pbar.set_postfix(train_loss=loss.item())
        with torch.no_grad():
            angle = rotation_error_deg_symmetry_aware(q_pred, q_gt, label, device)
            running_angle += angle.sum().item()

    

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

    if valid_epoch_loss < best_loss:
        best_loss = valid_epoch_loss
        torch.save(model.state_dict(), "pose_resnet_best.pth")
        print("Saved new best model (based on validation)")







"""from src.datasets.scene import LinemodSceneDataset
from src.datasets.scene import GTDetections
from src.datasets.resnet import ResNetDataset
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import random
import torch
from src.networks.resnet import ResNetQuat
from src.networks.resnet import SymmetryAwareGeodesicLoss, rotation_error_deg_symmetry_aware


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_root = "/content/6d-pose-estimation/data/Linemod_preprocessed"
batch_size = 32
num_epochs = 30
lr = 1e-4
scene_ds = LinemodSceneDataset(
        dataset_root=dataset_root,
        split="train",
        img_size=640,
    )
print("scene dataset pronto")
gt_detections = GTDetections(scene_ds)
print("detections pronte")
train_ds = ResNetDataset(
        scene_dataset=scene_ds,
        detection_provider=gt_detections,
        img_size=224,
        padding=0.1,
    )
print("resnet dataset pronto")
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
print("dataloader pronto")

model = ResNetQuat().to(device)

criterion = SymmetryAwareGeodesicLoss(device)

optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=7,   
    gamma=0.1      
)

best_val = float("inf")
patience = 4
bad_epochs = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_angle = 0.0

    for i, batch in enumerate(train_loader):
        rgb = batch["rgb"].to(device)
        gt_q = batch["rotation"].to(device)
        gt_l = batch["label"].to(device)

        pred_q = model(rgb)
        loss = criterion(pred_q, gt_q, gt_l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            angle = rotation_error_deg_symmetry_aware(pred_q, gt_q, gt_l, device)
            running_angle += angle.mean().item()

        if i % 50 == 0:
            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"Iter {i}/{len(train_loader)} "
                f"Loss: {loss.item():.4f} | "
                f"Angle: {angle.mean().item():.2f}°"
            )

    train_loss = running_loss / len(train_loader)
    train_angle = running_angle / len(train_loader)


    model.eval()
    val_loss = 0.0
    val_angle = 0.0

    with torch.no_grad():
        for batch in valid_loader:
            rgb = batch["rgb"].to(device)
            gt_q = batch["rotation"].to(device)
            gt_l = batch["label"].to(device)

            pred_q = model(rgb)
            loss = criterion(pred_q, gt_q, gt_l)

            val_loss += loss.item()
            val_angle += rotation_error_deg_symmetry_aware(pred_q, gt_q, gt_l, device).mean().item()

    val_loss /= len(valid_loader)
    val_angle /= len(valid_loader)

    print(
        f"===> Epoch {epoch+1} "
        f"Train loss: {train_loss:.4f} | "
        f"Train err: {train_angle:.2f}° || "
        f"Val loss: {val_loss:.4f} | "
        f"Val err: {val_angle:.2f}°"
    )
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Current LR: {current_lr:.2e}")
    scheduler.step()
    if val_loss < best_val:
        best_val = val_loss
        bad_epochs = 0
        torch.save(model.state_dict(), "best_pose.pth")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping")
            break
"""







