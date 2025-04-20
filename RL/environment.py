import gym
import torch
import numpy as np


class DetectionEnv(gym.Env):
    """RL Environment for object detection with two agents."""
    def __init__(self, dataset, feature_extractor, max_centers=50):
        super(DetectionEnv, self).__init__()
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.max_centers = max_centers
        self.current_idx = 0
        self.centers = []  # List of [x, y, class, conf]
        self.bboxes = []   # List of [x, y, w, h, conf_size]
        self.action_space_center = gym.spaces.Box(low=-1, high=1, shape=(4 + 20,))  # [Δx, Δy, conf, done, class_probs]
        self.action_space_size = gym.spaces.Box(low=-1, high=1, shape=(3,))        # [Δw, Δh, conf_size]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, 448, 448))
