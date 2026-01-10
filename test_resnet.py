import math
import torch
from enum import Enum
import torch.nn.functional as F

class SymmetryType(Enum):
    NONE = 0
    DISCRETE = 1
    AXIAL = 2


LINEMOD_SYMMETRIES = {
    3: SymmetryType.AXIAL,     # can
    7: SymmetryType.DISCRETE,  # eggbox
    8: SymmetryType.AXIAL,     # glue
    9: SymmetryType.DISCRETE,  # holepuncher (approx)
}

def rotate_vector(q, v):
    q_vec = q[:, 1:]
    w = q[:, :1]

    v = v.unsqueeze(0).expand_as(q_vec)
    t = 2 * torch.cross(q_vec, v, dim=1)
    v_rot = v + w * t + torch.cross(q_vec, t, dim=1)

    return F.normalize(v_rot, dim=1)



def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)


q_id = torch.tensor([1., 0., 0., 0.])
q_180_z = torch.tensor([0., 0., 0., 1.])

SYMMETRIC_QUATS = {
    10: torch.tensor([
        [1., 0., 0., 0.],
        [0., 0., 0., 1.],   # 180° z
    ]),
    12: torch.tensor([
        [1., 0., 0., 0.],
        [0., 0., 0., 1.],
    ]),
}



def axial_angle_deg(q_pred, q_gt, z_axis):
    z_pred = rotate_vector(q_pred.unsqueeze(0), z_axis)[0]
    z_gt   = rotate_vector(q_gt.unsqueeze(0), z_axis)[0]

    cos = torch.dot(z_pred, z_gt)
    cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)

    angle = torch.acos(cos)
    return torch.rad2deg(angle)


def discrete_angle_deg(q_pred, q_gt, label):
    angles = []

    for q_sym in SYMMETRIC_QUATS[label]:
        q_equiv = quat_mul(q_gt, q_sym)
        angles.append(geodesic_angle_deg(q_pred, q_equiv))

    return torch.min(torch.stack(angles))


def geodesic_angle_deg(q1, q2):
    dot = torch.sum(q1 * q2)
    dot = torch.abs(dot)
    dot = torch.clamp(dot, -1 + 1e-6, 1 - 1e-6)
    angle = 2 * torch.acos(dot)
    return torch.rad2deg(angle)

def rotation_error_deg_symmetry_aware(q_pred, q_gt, labels, device):
    z_axis = torch.tensor([0., 0., 1.], device=device)
    errors = []

    for i in range(q_pred.shape[0]):
        label = int(labels[i].item())
        sym = LINEMOD_SYMMETRIES.get(label, SymmetryType.NONE)

        if sym == SymmetryType.NONE:
            err = geodesic_angle_deg(q_pred[i], q_gt[i])

        elif sym == SymmetryType.DISCRETE:
            err = discrete_angle_deg(q_pred[i], q_gt[i], label)

        elif sym == SymmetryType.AXIAL:
            err = axial_angle_deg(q_pred[i], q_gt[i], z_axis)

        errors.append(err)

    return torch.stack(errors)


"""def quaternion_angular_error(q_pred, q_gt):
    dot = torch.sum(q_pred * q_gt, dim=1)
    dot = torch.clamp(torch.abs(dot), -1.0, 1.0)
    angle = 2 * torch.acos(dot)
    return angle * (180.0 / math.pi)"""

from torch.utils.data import DataLoader
from src.datasets.resnet import ResNetDataset
from src.datasets.scene import LinemodSceneDataset, GTDetections

scene_ds = LinemodSceneDataset("data/Linemod_preprocessed", split="test")
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

from src.models.resnet import PoseResNet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseResNet(pretrained=False)
model = model.to(device)
weight_path = "/content/drive/MyDrive/machine_learning_project/pose_resnet_best.pth"

try:
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(" Model weights loaded.")
except:
    print(" Loading with strict=False...")
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)



from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

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

