import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 2, 2); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1); self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1); self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1); self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        return x.view(x.size(0), -1)
    

class RGBDFusionNet(nn.Module):
    def __init__(self, pretrained=True):
        super(RGBDFusionNet, self).__init__()
 
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_resnet = models.resnet50(weights=weights)
        except:
            base_resnet = models.resnet50(pretrained=pretrained)

        self.rgb_encoder = nn.Sequential(*list(base_resnet.children())[:-1])
        self.depth_encoder = DepthNet()
        
        # Fusion & Heads
        self.fc1 = nn.Linear(2048 + 512, 1024)
        self.drop = nn.Dropout(0.3)
        
        # Output Head
        self.head = nn.Linear(1024, 4)  # Quaternion (4D)
       
    def forward(self, rgb, depth):
        f_rgb = self.rgb_encoder(rgb).view(rgb.size(0), -1)
        f_depth = self.depth_encoder(depth)
        
        # Concatenate features
        f_fused = torch.cat((f_rgb, f_depth), dim=1)
        
        x = F.relu(self.fc1(f_fused))
        x = self.drop(x)
        
        return F.normalize(self.head(x), p=2, dim=1)


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


