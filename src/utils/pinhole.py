import torch

def depth_to_points(depth, K, uv_grid):
    """
    depth:   [B, 1, H, W]
    uv_grid: [B, H, W, 2]
    K:       [3, 3]
    return:  [B, H, W, 3]
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = uv_grid[..., 0]
    v = uv_grid[..., 1]
    z = depth.squeeze(1)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return torch.stack([x, y, z], dim=-1)


def weighted_translation(points_3d, weights):
    """
    points_3d: [B, H, W, 3]
    weights:   [B, 1, H, W]
    """
    weights = weights.permute(0, 2, 3, 1)  # [B, H, W, 1]
    t = (points_3d * weights).sum(dim=(1,2))
    return t