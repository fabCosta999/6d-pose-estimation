import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)

class ResNetQuat(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feat_dim = 512

        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            feat_dim = 512

        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feat_dim = 2048

        else:
            raise ValueError("Unsupported backbone")

        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(feat_dim, 4)

    def forward(self, x):
        x = self.backbone(x)
        q = self.head(x)
        q = F.normalize(q, dim=1)
        return q


class QuaternionGeodesicLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q_pred, q_gt):
        dot = torch.sum(q_pred * q_gt, dim=1)
        dot = torch.abs(dot)
        return (1 - dot).mean()