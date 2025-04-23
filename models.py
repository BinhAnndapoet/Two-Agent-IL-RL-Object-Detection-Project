import torch
import torch.nn as nn

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