import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1); self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1); self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        return x.view(x.size(0), -1)
    

class DepthRotationNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthRotationNet, self).__init__()
 
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

