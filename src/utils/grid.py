import torch

def build_uv_grid(box, H, W, device):
    """
    box: [B, 4] -> (x, y, w, h) in pixel immagine
    return: uv_grid [B, H, W, 2]
    """
    B = box.shape[0]

    x, y, bw, bh = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

    i = torch.arange(H, device=device).float()
    j = torch.arange(W, device=device).float()
    ii, jj = torch.meshgrid(i, j, indexing="ij")

    ii = ii.unsqueeze(0).expand(B, -1, -1)
    jj = jj.unsqueeze(0).expand(B, -1, -1)

    u = x[:, None, None] + (jj + 0.5) * bw[:, None, None] / W
    v = y[:, None, None] + (ii + 0.5) * bh[:, None, None] / H

    return torch.stack([u, v], dim=-1)  # [B, H, W, 2]


def spatial_softmax(weight_map, mask=None, tau=0.05):
    B, _, H, W = weight_map.shape
    w = weight_map.view(B, -1) / tau

    if mask is not None:
        m = mask.view(B, -1)
        w = w.masked_fill(m == 0, -1e9)

    w = torch.softmax(w, dim=1)
    return w.view(B, 1, H, W)

def global_softmax(logits, tau=0.1):
    B, _, H, W = logits.shape
    w = logits.view(B, -1) / tau
    w = torch.softmax(w, dim=1)
    return w.view(B, 1, H, W)


def make_coord_grid(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=0)   # [2, H, W]
    return grid