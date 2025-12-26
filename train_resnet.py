from src.datasets.scene import LinemodSceneDataset
from src.datasets.scene import GTDetections
from src.datasets.resnet import ResNetDataset
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import random
import torch
from src.models.resnet import ResNetQuat
from src.models.resnet import QuaternionGeodesicLoss
from src.utils.quaternions import rotation_error_deg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_root = "/content/6d-pose-estimation/data/Linemod_preprocessed"
batch_size = 32
num_epochs = 50
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

model = ResNetQuat(
        backbone="resnet18",
        pretrained=True,
    ).to(device)

criterion = QuaternionGeodesicLoss()

optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_angle = 0.0

    for i, batch in enumerate(train_loader):
        rgb = batch["rgb"].to(device)
        gt_q = batch["rotation"].to(device)

        pred_q = model(rgb)
        loss = criterion(pred_q, gt_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            angle = rotation_error_deg(pred_q, gt_q)
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

            pred_q = model(rgb)
            loss = criterion(pred_q, gt_q)

            val_loss += loss.item()
            val_angle += rotation_error_deg(pred_q, gt_q).mean().item()

    val_loss /= len(valid_loader)
    val_angle /= len(valid_loader)

    print(
        f"===> Epoch {epoch+1} "
        f"Train loss: {train_loss:.4f} | "
        f"Train err: {train_angle:.2f}° || "
        f"Val loss: {val_loss:.4f} | "
        f"Val err: {val_angle:.2f}°"
    )
