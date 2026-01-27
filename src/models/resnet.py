import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class PoseResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(PoseResNet, self).__init__()
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
        except:
            self.model = models.resnet50(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 4)

    def forward(self, x):
        q = self.model(x)
        q = F.normalize(q, p=2, dim=1, eps=1e-6)
        return q
  
