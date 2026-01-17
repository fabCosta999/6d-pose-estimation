import torch


def entropy_loss_all(weights, eps=1e-8):
    B = weights.shape[0]
    w = weights.view(B, -1)
    entropy = -(w * torch.log(w + eps)).sum(dim=1)
    return entropy.mean()