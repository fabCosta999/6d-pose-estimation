import torch
import torch.nn.functional as F
from src.utils.linemod_symmetries import LINEMOD_SYMMETRIES, SymmetryType, SYMMETRIC_QUATS

def rotation_matrix_to_quaternion(R):
    trace = R.trace()

    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
    return q / torch.norm(q)  


def quaternion_to_rotation_matrix(q):
    q = q / q.norm()
    w, x, y, z = q
    return torch.tensor([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], device=q.device)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)



def geodesic_angle(q1, q2):
    q1 = F.normalize(q1, dim=-1, eps=1e-6)
    q2 = F.normalize(q2, dim=-1, eps=1e-6)

    dot = torch.abs(torch.sum(q1 * q2, dim=-1))
    dot = torch.clamp(dot, -1 + 1e-6, 1 - 1e-6)

    return 2 * torch.acos(dot)

    

def rotation_error_deg_symmetry_aware(q_pred, q_gt, labels, device):
    errors = []

    for i in range(q_pred.shape[0]):
        label = int(labels[i].item())
        sym = LINEMOD_SYMMETRIES.get(label, SymmetryType.NONE)

        if sym == SymmetryType.NONE:
            err = geodesic_angle_deg(q_pred[i], q_gt[i])

        elif sym == SymmetryType.DISCRETE:
            err = discrete_angle_deg(q_pred[i], q_gt[i], label)

        errors.append(err)

    return torch.stack(errors)


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