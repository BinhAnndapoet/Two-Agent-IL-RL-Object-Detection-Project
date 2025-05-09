import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

"""
    Architecture of the Vanilla (Standard) DQN model.
"""
class DQN(nn.Module):
    """
    The DQN network that estimates the action-value function

    Args:
        ninputs: The number of inputs
        noutputs: The number of outputs

    Layers:
        1. Linear layer with ninputs neurons
        2. ReLU activation function
        3. Dropout layer with 0.2 dropout rate
        4. Linear layer with 1024 neurons
        5. ReLU activation function
        6. Dropout layer with 0.2 dropout rate
        7. Linear layer with 512 neurons
        8. ReLU activation function
        9. Dropout layer with 0.2 dropout rate
        10. Linear layer with 256 neurons
        11. ReLU activation function
        12. Dropout layer with 0.2 dropout rate
        13. Linear layer with 128 neurons
        14. Output layer with noutputs neurons
    """
    def __init__(self, ninputs, noutputs):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(ninputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, noutputs)
        )

    def forward(self, X):
        # Forward pass
        return self.classifier(X)

    def __call__(self, X):
        return self.forward(X)


class Resnet18FeatureExtractor(nn.Module):
    """Extract multi-scale features from images using ResNet-18."""
    def __init__(self):
        super(Resnet18FeatureExtractor, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

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

        # x = self.pool(x)
        # x = self.flatten(x)

        return x

        # return {
        #     'low': low_level,    # List of low-level features
        #     'mid': mid_level,    # List of mid-level features
        #     'high': high_level   # List of high-level features
        # }


class ILModel(nn.Module):
    """
    Imitation Learning model for Center and Size Agents with multiple outputs.
    """
    def __init__(self, input_dim, phase="center", n_classes=20):
        super(ILModel, self).__init__()
        self.phase = phase
        self.n_classes = n_classes
        hidden_dim1 = 256
        hidden_dim2 = 128
        
        # Common backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        
        if self.phase == "center":
            # Outputs: (Δx, Δy), class_probs, conf, done
            self.pos_head = nn.Linear(hidden_dim2, 4)  # 4 actions: right, left, up, down
            self.class_head = nn.Linear(hidden_dim2, n_classes)  # 20 classes
            self.conf_head = nn.Linear(hidden_dim2, 1)  # Confidence
            self.done_head = nn.Linear(hidden_dim2, 1)  # Done signal
        else:
            # Outputs: (Δh, Δw), conf_size
            self.size_head = nn.Linear(hidden_dim2, 4)  # 4 actions: bigger, smaller, fatter, taller
            self.conf_head = nn.Linear(hidden_dim2, 1)  # Confidence

    def forward(self, x):
        x = self.backbone(x)
        if self.phase == "center":
            pos = self.pos_head(x)
            class_probs = self.class_head(x)
            conf = self.conf_head(x)
            done = self.done_head(x)
            return pos, class_probs, conf, done
        else:
            size = self.size_head(x)
            conf = self.conf_head(x)
            return size, conf
