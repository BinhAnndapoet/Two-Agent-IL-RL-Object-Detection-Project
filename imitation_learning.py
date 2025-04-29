import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from env import DetectionEnv
from models import ILModel
from utils import Transition

def select_expert_action_center(env, current_bbox, target_bbox):
    """
    Select optimal action for CenterDQNAgent based on ground truth.

    Args:
        env (DetectionEnv): Environment instance.
        current_bbox (list): Current bounding box [x1, y1, x2, y2].
        target_bbox (list): Ground truth bounding box [x1, y1, x2, y2].

    Returns:
        int: Optimal action (0: right, 1: left, 2: up, 3: down, 4: trigger).
    """
    current_center = [(current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2]
    target_center = [(target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2]
    dx = target_center[0] - current_center[0]
    dy = target_center[1] - current_center[1]
    alpha_w = env.alpha * (current_bbox[2] - current_bbox[0])
    alpha_h = env.alpha * (current_bbox[3] - current_bbox[1])

    # If center is close enough, trigger
    if abs(dx) < alpha_w / 2 and abs(dy) < alpha_h / 2:
        return 4  # Trigger

    # Choose action to minimize distance
    actions = [
        (0, dx > alpha_w),  # Right
        (1, dx < -alpha_w),  # Left
        (2, dy < -alpha_h),  # Up
        (3, dy > alpha_h)   # Down
    ]
    valid_actions = [a[0] for a in actions if a[1]]
    return random.choice(valid_actions) if valid_actions else 4


def select_expert_action_size(env, current_bbox, target_bbox):
    """
    Select optimal action for SizeDQNAgent based on ground truth.

    Args:
        env (DetectionEnv): Environment instance.
        current_bbox (list): Current bounding box [x1, y1, x2, y2].
        target_bbox (list): Ground truth bounding box [x1, y1, x2, y2].

    Returns:
        int: Optimal action (0: bigger, 1: smaller, 2: fatter, 3: taller, 4: trigger).
    """
    current_width = current_bbox[2] - current_bbox[0]
    current_height = current_bbox[3] - current_bbox[1]
    target_width = target_bbox[2] - target_bbox[0]
    target_height = target_bbox[3] - target_bbox[1]
    alpha_w = env.alpha * current_width
    alpha_h = env.alpha * current_height

    # If size is close enough, trigger
    if (abs(current_width - target_width) < alpha_w and 
        abs(current_height - target_height) < alpha_h):
        return 4  # Trigger

    # Choose action to adjust size
    actions = [
        (0, current_width < target_width and current_height < target_height),  # Bigger
        (1, current_width > target_width and current_height > target_height),  # Smaller
        (2, current_height > target_height),  # Fatter
        (3, current_width > target_width)     # Taller
    ]
    valid_actions = [a[0] for a in actions if a[1]]
    return random.choice(valid_actions) if valid_actions else 4
