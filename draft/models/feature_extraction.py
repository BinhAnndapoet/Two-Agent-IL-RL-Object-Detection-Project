import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

class FeatureExtractor(nn.Module):
    """Extract multi-scale features from images using ResNet-18."""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image [B, 3, H, W]
        Returns:
            dict: Multi-scale features
        """
        # Low-level features (edges, colors)
        low_level = []
        x = self.conv1(x)  # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        low_level.append(x)  # Low-level: edges, colors

        # Mid-level features (textures, patterns)
        mid_level = []
        x = self.maxpool(x)  # [B, 64, H/4, W/4]
        x = self.layer1(x)   # [B, 64, H/4, W/4]
        mid_level.append(x)  # Mid-level: basic textures
        x = self.layer2(x)   # [B, 128, H/8, W/8]
        mid_level.append(x)  # Mid-level: complex textures

        # High-level features (semantics)
        high_level = []
        x = self.layer3(x)   # [B, 256, H/16, W/16]
        high_level.append(x)
        x = self.layer4(x)   # [B, 512, H/32, W/32]
        high_level.append(x) # High-level: semantics

        return {
            'low': low_level,    # List of low-level features
            'mid': mid_level,    # List of mid-level features
            'high': high_level   # List of high-level features
        }

    # OPTIONAL
    # def extract_canny(self, img):
    #     """Extract Canny edges from image."""
    #     img_np = img.permute(1, 2, 0).cpu().numpy() * 255
    #     img_np = img_np.astype(np.uint8)
    #     edges = cv2.Canny(img_np, 100, 200)
    #     return torch.tensor(edges, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
