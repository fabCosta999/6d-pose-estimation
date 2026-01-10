import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


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
  
