import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoderWeightsNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- Encoder -----
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1); self.bn1 = nn.BatchNorm2d(16)
        
        self.down1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1); self.bn2 = nn.BatchNorm2d(32)
        self.down2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1); self.bn3 = nn.BatchNorm2d(64)

        self.down3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1); self.bn4 = nn.BatchNorm2d(128)

        # ----- Decoder -----
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1); self.bn5 = nn.BatchNorm2d(64)
        self.up2 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1); self.bn6 = nn.BatchNorm2d(32)
        self.up3 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1); self.bn7 = nn.BatchNorm2d(16)

        # Output: 1 unnormalized weight per pixel
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        x1 = F.relu(self.bn1(self.conv1(x)))    # 64x64x16
        x2 = F.relu(self.bn2(self.down1(x1)))   # 32x32x32
        x3 = F.relu(self.bn3(self.down2(x2)))   # 16x16x64
        x4 = F.relu(self.bn4(self.down3(x3)))   # 8x8x128

        # ----- Decoder -----
        u1 = F.relu(self.bn5(self.up1(x4)))     # 16x16x64
        u1 = torch.cat([u1, x3], dim=1)         # 16x16x128
        u2 = F.relu(self.bn6(self.up2(u1)))     # 32x32x32
        u2 = torch.cat([u2, x2], dim=1)         # 32x32x64
        u3 = F.relu(self.bn7(self.up3(u2)))     # 64x64x16
        w = self.out_conv(u3)                   # 64x64x1
        return w
