import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.grid import spatial_softmax, build_uv_grid
from src.utils.pinhole import weighted_translation, depth_to_points

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
    
class DepthTranslationNet(nn.Module):
    def __init__(self, depth_mean, depth_std):
        super().__init__()
        self.end_dec = EncoderDecoderWeightsNet()
        
        self.register_buffer('depth_mean', torch.tensor(depth_mean))
        self.register_buffer('depth_std', torch.tensor(depth_std))

    def forward(self, rgb, depth, coord, box, K):
        # 1. depth denormalization
        un_normalized_depth = depth * self.depth_std + self.depth_mean

        # 2. encoder-decoder input
        x = torch.cat([rgb, depth, coord], dim=1)

        # 3. encoder-decoder output
        logits = self.end_dec(x)  

        # 4. pixel < 10 mm are considered backgroud
        valid_mask = (un_normalized_depth > 10.0).float()
        
        # 5 weights computation
        weights = spatial_softmax(logits, valid_mask)

        # 4. Ricostruzione 3D (Inverse Pinhole)
        B, _, H, W = depth.shape
        device = depth.device
        
        # 5. u-v grid creation based on bounding box crop
        uv_grid = build_uv_grid(box, H, W, device)

        # 6. 2D -> 3D points projection
        points_3d = depth_to_points(un_normalized_depth, K, uv_grid)

        # 7. translation regression
        t_pred = weighted_translation(points_3d, weights)

        return weights, t_pred