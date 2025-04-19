import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterAgent(nn.Module):
    """Center Agent for predicting bounding box centers and class labels."""
    def __init__(self, num_classes=20, feature_channels=512):
        """
        Args:
            num_classes (int): Number of classes (default: 20 for PASCAL VOC).
            feature_channels (int): Number of feature channels from FeatureExtractor (default: 512 for layer4).
        """
        super(CenterAgent, self).__init__()
        self.num_classes = num_classes
        self.feature_channels = feature_channels
        self.conv = nn.Sequential(
            nn.Conv2d(feature_channels + 3 + 1, 128, kernel_size=3, padding=1),  # +1 for center map
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Adjust FC layer size based on input image size (448x448 -> 28x28 after conv)
        self.fc = nn.Sequential(
            nn.Linear(16 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 4 + num_classes)  # [Δx, Δy, conf, done, class_probs]
        )

    def _create_center_map(self, centers, height, width):
        """
        Create a spatial map of current centers.
        Args:
            centers (torch.Tensor): [B, N, 4] (x, y, class, conf)
            height (int): Image height
            width (int): Image width
        Returns:
            torch.Tensor: Center map [B, 1, H, W]
        """
        batch_size = centers.size(0)
        center_map = torch.zeros(batch_size, 1, height, width, device=centers.device)
        for b in range(batch_size):
            for n in range(centers.size(1)):
                x, y, _, conf = centers[b, n]
                x_pixel = int(x * width)
                y_pixel = int(y * height)
                if 0 <= x_pixel < width and 0 <= y_pixel < height:
                    center_map[b, 0, y_pixel, x_pixel] = conf  # Use confidence as intensity
        return center_map   

    def forward(self, image, features, centers):
        """
        Args:
            image (torch.Tensor): Input image [B, 3, H, W]
            features (torch.Tensor): High-level features from layer4 [B, 512, H/32, W/32]
            centers (torch.Tensor): Current centers [B, N, 4] (x, y, class, conf)
        Returns:
            torch.Tensor: Action [B, 4 + num_classes] (Δx, Δy, conf, done, class_probs)
        """
        # Resize features to match image size
        features = F.interpolate(features, size=image.shape[2:], mode='bilinear', align_corners=False)
        
        # Create center map from centers
        center_map = self._create_center_map(centers, image.shape[2], image.shape[3])
        
        # Concatenate image, features, and center map
        x = torch.cat([image, features, center_map], dim=1)  # [B, 516, H, W]
        
        # Process through conv and fc layers
        x = self.conv(x)  # [B, 16, H/16, W/16]
        x = x.view(x.size(0), -1)  # Flatten
        action = self.fc(x)
        
        # Apply sigmoid to conf and done
        action[:, 2] = torch.sigmoid(action[:, 2])  # conf
        action[:, 3] = torch.sigmoid(action[:, 3])  # done
        return action

    def imitate(self, image, features, centers, target_action):
        """
        Imitation Learning for Center Agent.
        Args:
            image (torch.Tensor): Input image [B, 3, H, W]
            features (torch.Tensor): High-level features [B, 512, H/32, W/32]
            centers (torch.Tensor): Current centers [B, N, 4]
            target_action (torch.Tensor): [B, 4 + num_classes] (Δx, Δy, conf, done, class_probs)
        Returns:
            torch.Tensor: Loss
        """
        action = self.forward(image, features, centers)
        loss_pos = F.mse_loss(action[:, :2], target_action[:, :2])  # Δx, Δy
        loss_conf = F.mse_loss(action[:, 2], target_action[:, 2])  # Confidence
        loss_done = F.binary_cross_entropy(action[:, 3], target_action[:, 3])  # Done
        loss_class = F.cross_entropy(action[:, 4:], target_action[:, 4:].argmax(dim=1))  # Class
        total_loss = loss_pos + loss_conf + loss_done + loss_class
        return total_loss