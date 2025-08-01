# --- START OF FILE models/networks.py ---
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResNet18FeatureExtractor(nn.Module):
    """Extract features from images using ResNet-18."""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Exclude the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image [B, 3, H, W]
        Returns:
            torch.Tensor: Flattened feature vector [B, 512]
        """
        x = self.features(x)
        x = self.flatten(x)
        return x

class ILModel(nn.Module):
    """Multi-head network for Imitation Learning."""
    def __init__(self, input_dim, phase="center", n_classes=20):
        super().__init__()
        self.phase = phase
        self.n_classes = n_classes
        hidden_dim1, hidden_dim2, hidden_dim3 = 256, 128, 64

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3), nn.ReLU()
        )

        if self.phase == "center":
            self.pos_head = nn.Linear(hidden_dim3, 4)
            self.class_head = nn.Linear(hidden_dim3, n_classes) 
            self.conf_head = nn.Linear(hidden_dim3, 1)
            self.done_head = nn.Linear(hidden_dim3, 1)
        else: # "size"
            self.size_head = nn.Linear(hidden_dim3, 4)
            self.conf_head = nn.Linear(hidden_dim3, 1)

    def forward(self, x):
        x = self.backbone(x)
        if self.phase == "center":
            return self.pos_head(x), self.class_head(x), self.conf_head(x), self.done_head(x)
        else:
            return self.size_head(x), self.conf_head(x)

class DQN(nn.Module):
    """Multi-head Q-network for DQN Agent. Same architecture as IL for weight transfer."""
    def __init__(self, input_dim, n_outputs, phase="center", n_classes=20):
        super().__init__()
        self.phase = phase
        self.n_classes = n_classes
        hidden_dim1, hidden_dim2, hidden_dim3 = 256, 128, 64

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3), nn.ReLU()
        )

        if self.phase == "center":
            # The action space is split into position, class, confidence, and done
            self.pos_head = nn.Linear(hidden_dim3, 4)
            self.class_head = nn.Linear(hidden_dim3, n_classes)
            self.conf_head = nn.Linear(hidden_dim3, 1)
            self.done_head = nn.Linear(hidden_dim3, 1)
        else: # "size"
            self.size_head = nn.Linear(hidden_dim3, 4)
            self.conf_head = nn.Linear(hidden_dim3, 1)

    def forward(self, x):
        x = self.backbone(x)
        if self.phase == "center":
            return self.pos_head(x), self.class_head(x), self.conf_head(x), self.done_head(x)
        else:
            return self.size_head(x), self.conf_head(x)
# --- END OF FILE models/networks.py ---