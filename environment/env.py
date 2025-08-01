# --- START OF FILE environment/env.py ---
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch

from models.networks import ResNet18FeatureExtractor
from utils.helpers import transform_input
from environment.rewards import calculate_reward, calculate_trigger_reward
from utils.metrics import calculate_best_iou
from environment.rendering import render, display, generate_random_color

class DetectionEnv(gym.Env):
    def __init__(self, env_config: dict):
        super().__init__()
        # Unpack config
        for key, value in env_config.items():
            setattr(self, key, value)
        
        self.class_names = self.current_class

        # Init state variables
        self.image = None
        self.original_image = None
        self.width, self.height = self.target_size
        self.bbox = [0, 0, 0, 0]
        self.current_gt_bboxes, self.current_gt_labels = [], []
        self.current_gt_index, self.step_count, self.trigger_count = 0, 0, 0
        self.done, self.truncated = False, False
        self.class_image_index = 0
        self.detected_centers = []
        self.no_of_triggers = 0
        self.target_bbox = [0, 0, 0, 0]
        self.color = (255, 0, 0)
        self.action_history_center = [[0] * (self.number_actions + self.n_classes) for _ in range(self.action_history_size)]
        self.action_history_size_list = [[0] * self.number_actions for _ in range(self.action_history_size)]
        
        self.feature_extractor = ResNet18FeatureExtractor().to(self.device)
        self.classification_dictionary = {"label": [], "confidence": [], "bbox": [], "color": [], "aspect": []}
        self.detected_objects = set()

        self.iterators = {}
        if self.dataset:
            for split in ['train_il', 'train_dqn', 'test']:
                if split in self.dataset:
                    self.iterators[split] = iter(self.dataset[split][1])
        
        self._update_spaces()

    def _update_spaces(self):
        if self.phase == "center":
            total_center_actions = self.number_actions + self.n_classes
            self.action_space = spaces.Tuple((spaces.Discrete(self.number_actions), spaces.Discrete(self.n_classes)))
            history_dim = self.action_history_size * total_center_actions
        else: # size
            self.action_space = spaces.Discrete(self.number_actions)
            history_dim = self.action_history_size * self.number_actions
        
        state_dim = self.feature_dim + 2 + history_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

    def get_class_names(self):
        return self.current_class
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.extract_data() # Load new image
        self.height, self.width = self.image.shape[:2]
        
        # Reset state
        self.current_gt_index = np.random.randint(0, len(self.current_gt_bboxes)) if self.current_gt_bboxes else 0
        self.target_bbox = self.current_gt_bboxes[self.current_gt_index] if self.current_gt_bboxes else [0, 0, 0, 0]
        self.bbox = [0, 0, int(self.width * 0.75), int(self.height * 0.75)] # Simple starting box
        self.phase = "center"
        self._update_spaces()
        self.step_count = self.trigger_count = self.no_of_triggers = 0
        self.done = self.truncated = False
        self.color = generate_random_color()
        self.classification_dictionary = {"label": [], "confidence": [], "bbox": [], "color": [], "aspect": []}
        self.detected_objects.clear()
        self.detected_centers.clear()
        
        return self.get_state(), self.get_info()

    def step(self, action):
        previous_bbox = self.bbox.copy()
        
        if self.phase == "center":
            pos_action, class_action = action
            if pos_action < 4: # move
                self.transform_action(pos_action)
            elif pos_action == 4: # trigger
                iou = calculate_best_iou([self.bbox], self.current_gt_bboxes)
                if iou >= self.threshold:
                    self.phase = "size"
                    self._update_spaces()
        else: # size
            if action < 4: # resize
                self.transform_action(action)
            elif action == 4: # trigger
                iou = calculate_best_iou([self.bbox], self.current_gt_bboxes)
                if iou >= self.threshold:
                    self.restart_and_change_state()

        reward = calculate_reward(self, self.bbox, previous_bbox, action)
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.truncated = True
        if self.no_of_triggers >= len(self.current_gt_bboxes) and self.current_gt_bboxes:
            self.done = True

        return self.get_state(), reward, self.done, self.truncated, self.get_info()

    def get_state(self):
        x1, y1, x2, y2 = map(int, self.bbox)
        cropped_image = self.image[y1:y2, x1:x2] if self.image is not None and y2 > y1 and x2 > x1 else np.zeros((1,1,3), dtype=np.uint8)
        
        image_tensor = transform_input(cropped_image, self.target_size)
        with torch.no_grad():
            features = self.feature_extractor(image_tensor.unsqueeze(0).to(self.device)).cpu().numpy().flatten()
        
        centers = np.array([(x1 + x2) / 2 / self.width, (y1 + y2) / 2 / self.height])
        
        if self.phase == "center":
            history = np.array(self.action_history_center).flatten()
        else:
            history = np.array(self.action_history_size_list).flatten()
            
        return np.concatenate([features, centers, history]).astype(np.float32)

    def extract_data(self):
        # Logic to get next image from dataloader...
        data_split = 'train_dqn' if self.phase in ['dqn', 'test'] else 'train_il'
        try:
            _, imgs, bboxes, labels = next(self.iterators[data_split])
            idx = self.class_image_index % len(imgs)
            img_tensor = imgs[idx]
            self.image = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            self.original_image = self.image.copy()
            self.current_gt_bboxes = bboxes[idx]
            self.current_gt_labels = labels[idx]
            self.class_image_index += 1
        except StopIteration:
            self.iterators[data_split] = iter(self.dataset[data_split][1])
            self.extract_data() # Retry

    def get_info(self):
        return {"phase": self.phase, "iou": calculate_best_iou([self.bbox], self.current_gt_bboxes), "gt_bboxes": self.current_gt_bboxes}
    
    def transform_action(self, action):
        x1, y1, x2, y2 = self.bbox
        w, h = x2 - x1, y2 - y1
        dx, dy = int(self.alpha * w), int(self.alpha * h)
        # Bbox transform logic...
        self.bbox = [max(0, x1), max(0, y1), min(self.width, x2), min(self.height, y2)]
    
    def restart_and_change_state(self):
        self.classification_dictionary["bbox"].append(self.bbox)
        self.classification_dictionary["color"].append(self.color)
        self.no_of_triggers += 1
        self.detected_objects.add(self.current_gt_index)
        
        remaining = [i for i in range(len(self.current_gt_bboxes)) if i not in self.detected_objects]
        if not remaining:
            self.done = True
            return

        self.current_gt_index = random.choice(remaining)
        self.target_bbox = self.current_gt_bboxes[self.current_gt_index]
        self.phase = "center"
        self._update_spaces()
        self.bbox = [0, 0, int(self.width * 0.75), int(self.height * 0.75)] # Reset bbox
        self.color = generate_random_color()

    def render(self, mode="rgb_array"):
        return render(self, mode)

    def display(self, mode):
        return display(self, mode)
# --- END OF FILE environment/env.py ---