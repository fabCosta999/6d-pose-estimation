import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.datasets.rgbd import RGBDDataset
from src.datasets.scene import LinemodSceneDataset, GTDetections
from src.models.rgbd import RGBDFusionNet
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from src.utils.quaternions import rotation_error_deg_symmetry_aware


scene_ds = LinemodSceneDataset("data/Linemod_preprocessed", split="test")
dp = GTDetections(scene_ds)
test_ds = RGBDFusionNet(
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
model = RGBDFusionNet(pretrained=False)
model = model.to(device)
weight_path = "/content/drive/MyDrive/machine_learning_project/pose_rgbd_best.pth"

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
        depth = batch["depth"].to(device)
        labels = batch["label"]  # CPU ok

        q_pred = model(rgb, depth)

        errors = rotation_error_deg_symmetry_aware(q_pred, q_gt, labels, device)

        for err, lbl in zip(errors.cpu().numpy(), labels.numpy()):
            errors_per_class[int(lbl)].append(err)
print("\n" + "="*60)
print("POSE ESTIMATION RESULTS (per class)")
print("="*60)

all_errors = []

for cls, errs in errors_per_class.items():
    errs = np.array(errs)
    all_errors.extend(errs.tolist())

    mean_err = errs.mean()
    median_err = np.median(errs)
    acc_5 = np.mean(errs < 5) * 100
    acc_10 = np.mean(errs < 10) * 100
    acc_20 = np.mean(errs < 20) * 100

    print(f"\nClass {cls}:")
    print(f"  Samples: {len(errs)}")
    print(f"  Mean error:   {mean_err:.2f}°")
    print(f"  Median error: {median_err:.2f}°")
    print(f"  Acc < 5°:  {acc_5:.2f}%")
    print(f"  Acc < 10°: {acc_10:.2f}%")
    print(f"  Acc < 20°: {acc_20:.2f}%")

all_errors = np.array(all_errors)

print("\n" + "-"*60)
print("OVERALL:")
print(f"  Mean error:   {all_errors.mean():.2f}°")
print(f"  Median error: {np.median(all_errors):.2f}°")
print(f"  Acc < 10°: {np.mean(all_errors < 10)*100:.2f}%")
print(f"  Acc < 20°: {np.mean(all_errors < 20)*100:.2f}%")
print("="*60)

