import torch
import torch.nn as nn
import torch.nn.functional as F

class SizeAgent(nn.Module):
    """Size Agent for adjusting bounding box width and height."""
    def __init__(self, feature_channels=512):
        """
        Args:
            feature_channels (int): Number of feature channels from FeatureExtractor (default: 512 for layer4).
        """
        super(SizeAgent, self).__init__()
        self.feature_channels = feature_channels
        self.conv = nn.Sequential(
            nn.Conv2d(feature_channels + 3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Adjust FC layer size based on patch size (64x64 -> 4x4 after conv)
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # [Δw, Δh, conf_size]
        )

    def forward(self, image_patch, features_patch):
        """
        Args:
            image_patch (torch.Tensor): Local image patch [B, 3, 64, 64]
            features_patch (torch.Tensor): Local features from layer4 [B, 512, 4, 4]
        Returns:
            torch.Tensor: Action [B, 3] (Δw, Δh, conf_size)
        """
        # Resize features to match image patch size
        features_patch = F.interpolate(features_patch, size=(64, 64), mode='bilinear', align_corners=False)
        x = torch.cat([image_patch, features_patch], dim=1)  # [B, 515, 64, 64]
        x = self.conv(x)  # [B, 32, 4, 4]
        x = x.view(x.size(0), -1)  # Flatten
        action = self.fc(x)
        # Apply sigmoid to conf_size
        action[:, 2] = torch.sigmoid(action[:, 2])  # conf_size
        return action

    def imitate(self, image_patch, features_patch, target_action):
        """
        Imitation Learning for Size Agent.
        Args:
            image_patch (torch.Tensor): Local image patch [B, 3, 64, 64]
            features_patch (torch.Tensor): Local features [B, 512, 4, 4]
            target_action (torch.Tensor): [B, 3] (Δw, Δh, conf_size)
        Returns:
            torch.Tensor: Loss
        """
        action = self.forward(image_patch, features_patch)
        loss_size = F.mse_loss(action[:, :2], target_action[:, :2])  # Δw, Δh
        loss_conf = F.mse_loss(action[:, 2], target_action[:, 2])   # Confidence
        total_loss = loss_size + loss_conf
        return total_loss