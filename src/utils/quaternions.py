import torch

def rotation_matrix_to_quaternion(R):
    """
    R: tensor (3, 3)
    ritorna: tensor (4,) [qw, qx, qy, qz]
    """
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
    return q / torch.norm(q)  # normalizzazione fondamentale


"""def rotation_error_deg(q_pred, q_gt):
    dot = torch.sum(q_pred * q_gt, dim=1)
    dot = torch.abs(dot)
    dot = torch.clamp(dot, -1 + 1e-6, 1 - 1e-6)
    angle = 2 * torch.acos(dot)
    return torch.rad2deg(angle)"""



