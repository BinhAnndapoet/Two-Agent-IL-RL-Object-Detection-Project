import os
import torch
import random
import numpy as np
import pandas as pd
import cv2
from collections import namedtuple, deque

# hyperparameters
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BUFFER_SIZE             = 0
BATCH_SIZE              = 0
NUM_IL_TRAJECTORIES     = 0
NUM_IL_EPOCHS           = 0
REPLAY_BUFFER_CAPACITY  = 0
TARGET_UPDATE_FREQ      = None
EXPLORATION_MODE        = None
ALPHA                   = 0
EPS_START               = 0
GUIDED_EXPLORE          = 0
GAMMA                   = 0
EPS_END                 = 0
EPS_DECAY               = 0
SUCCESS_CRITERIA_EPS    = 0
NU                      = 0
THRESHOLD               = 0
MAX_STEPS               = 0
TRIGGER_STEPS           = 0
NUMBER_ACTIONS          = 0
ACTION_HISTORY_SIZE     = 0
OBJ_CONFIG              = None
N_CLASSES               = 0
TARGET_SIZE             = (0, 0)
FEATURE_DIM             = 0
USE_DATASET             = True
ENV_MODE                = None
EPOCHS                  = 0
CURRENT_CLASSES         = 0
WINDOW_SIZE             = 0
DATASET                 = 0

env_config = {
    "dataset" : DATASET,
    "alpha" : ALPHA,
    "nu" : NU,
    "threshold" : THRESHOLD,
    "max_steps" : MAX_STEPS,
    "trigger_steps" : TRIGGER_STEPS,
    "number_actions" : NUMBER_ACTIONS,
    "action_history_size" : ACTION_HISTORY_SIZE,
    "object_config" : OBJ_CONFIG,
    "n_classes" : N_CLASSES,
    "target_size" : TARGET_SIZE,
    "feature_dim" : FEATURE_DIM,
    "device" : DEVICE,
    "use_dataset" : USE_DATASET,
    "env_mode" : ENV_MODE,
    "epochs" : EPOCHS,
    "current_class" : CURRENT_CLASSES,
    'window_size' : CURRENT_CLASSES
}



# Transition named tuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'next_state'))

class ReplayBuffer():
    """
    Replay buffer for storing and sampling transitions.

    Args:
        capacity (int): Maximum number of transitions to store.
    """
    def __init__(self, capacity=BUFFER_SIZE, batch_size=BATCH_SIZE):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size

    def append(self, transition):
        """
        Add a transition to the buffer.

        Args:
            transition (Transition): Transition to store.
        """
        self.memory.append(transition)

    def sample_batch(self):
        """
        Sample a batch of transitions.

        Returns:
            tuple: Batched (states, actions, rewards, dones, next_states).
        """
        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*batch))
        states = torch.from_numpy(np.array(batch.state, dtype=np.float32))
        actions = torch.from_numpy(np.array(batch.action, dtype=np.int64)).unsqueeze(1)
        rewards = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).unsqueeze(1)
        dones = torch.from_numpy(np.array(batch.done, dtype=np.bool8)).unsqueeze(1).to(torch.bool)
        next_states = torch.from_numpy(np.array(batch.next_state, dtype=np.float32))
        return states, actions, rewards, dones, next_states
    
    def __len__(self):
        return len(self.memory)


def transform_input(image, target_size):
    """
    Transform image for feature extraction.

    Args:
        image (np.ndarray): Input image.
        target_size (tuple): Target size (width, height).

    Returns:
        torch.Tensor: Transformed image tensor.
    """
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image, dtype=torch.float32)

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.

    Args:
        box1 (list): First box [x1, y1, x2, y2].
        box2 (list): Second box [x1, y1, x2, y2].

    Returns:
        float: IoU value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def calculate_best_iou(pred_boxes, gt_boxes):
    """
    Calculate the best IoU for predicted boxes against ground truth.

    Args:
        pred_boxes (list): Predicted boxes.
        gt_boxes (list): Ground truth boxes.

    Returns:
        float: Maximum IoU.
    """
    return max(calculate_iou(pred, gt) for pred in pred_boxes for gt in gt_boxes)
