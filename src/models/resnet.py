import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch


class PoseResNet(nn.Module):
    """
    ResNet50-based architecture for 6D Pose Estimation (Rotation only).
    Replaces the final classification head with a regression head for Quaternions.
    """
    def __init__(self, pretrained=True):
        super(PoseResNet, self).__init__()
        
        # Load pre-trained ResNet50
        # 'weights="DEFAULT"' is the modern way to load ImageNet weights in newer PyTorch versions
        # If using older PyTorch, use pretrained=True
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
        except:
            self.model = models.resnet50(pretrained=pretrained)
        
        # Modify the fully connected layer
        # ResNet50's output features are 2048. We need 4 (w, x, y, z).
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 4)

    def forward(self, x):
        # Forward pass
        q = self.model(x)
        
        # Normalize quaternion to ensure valid rotation (unit length)
        # This is crucial for geometric stability.
        q = F.normalize(q, p=2, dim=1, eps=1e-6)
        return q
  



"""class QuaternionLoss(nn.Module):
    def forward(self, q_pred, q_gt):
        dot = torch.sum(q_pred * q_gt, dim=1)
        loss = 1.0 - torch.abs(dot)
        return loss.mean()"""


from enum import Enum

class SymmetryType(Enum):
    NONE = 0
    DISCRETE = 1
    AXIAL = 2


LINEMOD_SYMMETRIES = {
    5:  SymmetryType.AXIAL,     # can
    10: SymmetryType.DISCRETE,  # eggbox
    11: SymmetryType.AXIAL,     # glue
    12: SymmetryType.DISCRETE,  # holepuncher (approx)
}


def rotate_vector(q, v):
    """
    q: (B, 4)
    v: (3,)
    """
    q_vec = q[:, 1:]
    w = q[:, :1]

    v = v.unsqueeze(0).expand_as(q_vec)
    t = 2 * torch.cross(q_vec, v, dim=1)
    v_rot = v + w * t + torch.cross(q_vec, t, dim=1)

    return F.normalize(v_rot, dim=1, eps=1e-6)



def quat_mul(q1, q2):
    """
    q1, q2: (..., 4)  [w, x, y, z]
    """
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

def geodesic_angle(q1, q2):
    """
    q1, q2: (..., 4)
    return: angolo in radianti
    """
    q1 = F.normalize(q1, dim=-1, eps=1e-6)
    q2 = F.normalize(q2, dim=-1, eps=1e-6)

    dot = torch.abs(torch.sum(q1 * q2, dim=-1))
    dot = torch.clamp(dot, -1 + 1e-6, 1 - 1e-6)

    return 2 * torch.acos(dot)


    

class SymmetryAwareGeodesicLoss(nn.Module):
    def __init__(self, device, axial_weight=0.1):
        super().__init__()
        self.device = device
        self.z_axis = torch.tensor([0., 0., 1.], device=device)
        self.axial_weight = axial_weight

    def forward(self, q_pred, q_gt, labels):
        """
        q_pred, q_gt: (B, 4)
        labels: (B,)
        """
        q_pred = F.normalize(q_pred, dim=1, eps=1e-6)
        q_gt   = F.normalize(q_gt, dim=1, eps=1e-6)

        losses = []

        for i in range(q_pred.size(0)):
            label = int(labels[i].item())
            sym = LINEMOD_SYMMETRIES.get(label, SymmetryType.NONE)

            # ---------------- NONE ----------------
            if sym == SymmetryType.NONE:
                loss = geodesic_angle(q_pred[i], q_gt[i])

            # ---------------- DISCRETE ----------------
            elif sym == SymmetryType.DISCRETE:
                q_syms = SYMMETRIC_QUATS[label].to(self.device)
                q_gt_syms = quat_mul(
                    q_gt[i].unsqueeze(0).expand(len(q_syms), -1),
                    q_syms
                )
                loss = torch.min(geodesic_angle(
                    q_pred[i].unsqueeze(0).expand_as(q_gt_syms),
                    q_gt_syms
                ))

            # ---------------- AXIAL ----------------
            elif sym == SymmetryType.AXIAL:
                z_pred = rotate_vector(q_pred[i:i+1], self.z_axis)[0]
                z_gt   = rotate_vector(q_gt[i:i+1], self.z_axis)[0]

                loss = 1 - torch.dot(z_pred, z_gt)

            losses.append(loss)

        return torch.stack(losses).mean()

    

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


