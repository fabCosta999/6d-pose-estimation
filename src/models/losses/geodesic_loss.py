import torch
import torch.nn.functional as F
from src.utils.linemod_symmetries import LINEMOD_SYMMETRIES, SymmetryType, SYMMETRIC_QUATS
from src.utils.quaternions import geodesic_angle, quat_mul, rotate_vector


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
            if sym == SymmetryType.NONE:
                loss = geodesic_angle(q_pred[i], q_gt[i])
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
            losses.append(loss)
        return torch.stack(losses).mean()