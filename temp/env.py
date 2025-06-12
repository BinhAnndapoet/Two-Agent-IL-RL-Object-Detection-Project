"""
    Custom Gymnasium environments for two-agent sequential object detection using Imitation Learning + Deep Q-Learning.

    Modules:
    - calculate_iou: utility for computing IoU between two bounding boxes.
    - DetectionEnv: environment for Center and Size Agents.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from utils import calculate_best_iou, calculate_best_recall, transform_input
from models import ResNet18FeatureExtractor
import torch
import random
import colorsys
import pygame

class DetectionEnv(gym.Env):
    def __init__(self, env_config: dict):
        super().__init__()
        # Config variables
        self.dataset = env_config.get("dataset", [])
        self.alpha = env_config.get("alpha", 0.1)
        self.nu = env_config.get("nu", 0.5)
        self.threshold = env_config.get("threshold", 0.6)
        self.max_steps = env_config.get("max_steps", 200)
        self.trigger_steps = env_config.get("trigger_steps", 10)
        self.number_actions = env_config.get("number_actions", 6)  # 4 move + trigger + conf
        self.action_history_size = env_config.get("action_history_size", 7)
        self.obj_config = env_config.get("object_config", "MULTI_OBJECT")
        self.n_classes = env_config.get("n_classes", 20)
        self.target_size = env_config.get("target_size", (448, 448))
        self.feature_dim = env_config.get("feature_dim", 512)
        self.device = env_config.get("device", "cpu")
        self.use_dataset = env_config.get("use_dataset", True)
        self.env_mode = env_config.get("env_mode", 0)
        self.epochs = env_config.get("epochs", 100)
        self.current_class = env_config.get("current_class", None)
        self.window_size = env_config.get("window_size", (448, 448))
        self.metadata = {"render_fps": 30}
        self.clock = pygame.time.Clock()

        # Environment state variables
        self.phase = "center"
        self.image = None
        self.width = 448
        self.height = 448
        self.bbox = [0, 0, 0, 0]
        self.current_gt_bboxes = []
        self.current_gt_labels = []
        self.current_gt_index = 0
        self.step_count = 0
        self.trigger_count = 0
        self.cumulative_reward = 0
        self.done = False
        self.truncated = False
        self.class_image_index = 0
        self.evaluation_results = {'gt_boxes': {}, 'gt_labels': {}}
        self.current_class = None
        self.detected_centers = []
        self.current_action = None

        # Action history
        self.action_history_center = [[0] * (self.number_actions + self.n_classes) for _ in range(self.action_history_size)]
        self.action_history_size_list = [[0] * self.number_actions for _ in range(self.action_history_size)] # [change]: self.action_history_size -> self.action_history_size_list
        self.render_mode = "rgb_array"
        self.feature_extractor = ResNet18FeatureExtractor().to(self.device)
        self.classification_dictionary = {"label": [], "confidence": [], "bbox": [], "color": [], "aspect": []}
        self.class_names = [f"class_{i}" for i in range(self.n_classes)]
        self.detected_objects = set()

        self._update_spaces()

    def _update_spaces(self):
        # [change]: update state_dim from list to int (line 81, 85) because of initializing incorrectly action_history_size & rename "Center" phase to "center"
        if self.phase == "center":
            self.action_space = spaces.MultiDiscrete([4, self.n_classes])
            history_dim = self.action_history_size * (self.number_actions + self.n_classes) # 7 * (6 + 20) = 182
            state_dim = self.feature_dim + 2 + history_dim # 512 + 2 + 182 = 696
        else:
            self.action_space = spaces.Discrete(self.number_actions)
            history_dim = self.action_history_size * self.number_actions # 7 * 6 = 42
            state_dim = self.feature_dim + 2 + history_dim # [update]: 512 + 2 + 42 = 556 -> add center to size state
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )

    def get_current_gt_bboxes(self):
        return self.current_gt_bboxes

    def set_current_gt_bboxes(self, value):
        self.current_gt_bboxes = value if isinstance(value, list) else [value]

    def get_class_names(self):
        return self.class_names

    def calculate_reward(self, current_state, previous_state, target_bboxes, target_labels, phase="center", action=None):
        self.phase = phase
        if not target_bboxes or not target_labels:
            return -1.0  # [update]: Trả về thưởng âm nếu không có mục tiêu
        
        
        if self.phase == "center":
            # Tính R_center
            cx_new = (current_state[0] + current_state[2]) / 2
            cy_new = (current_state[1] + current_state[3]) / 2
            cx_prev = (previous_state[0] + previous_state[2]) / 2
            cy_prev = (previous_state[1] + previous_state[3]) / 2
            dis_new = min(np.hypot(cx_new - (b[0] + b[2]) / 2, cy_new - (b[1] + b[3]) / 2) for b in target_bboxes) if target_bboxes else float('inf')
            dis_prev = min(np.hypot(cx_prev - (b[0] + b[2]) / 2, cy_prev - (b[1] + b[3]) / 2) for b in target_bboxes) if target_bboxes else float('inf')
            R_center = 1.0 if dis_new < dis_prev else -1.0

            # Tính R_class
            R_class = 0.0
            if action is not None and isinstance(action, (list, tuple)) and len(action) > 1:
                class_action = action[1]
                pred_class = self.class_names[class_action]
                gt_class = target_labels[self.current_gt_index] if target_labels and self.current_gt_index < len(target_labels) else "unknown" # [update]: add more condition to handle case when gt_labels index is out of range
                R_class = 1.0 if pred_class == gt_class else -1.0

            # Tính R_conf
            R_conf = 0.0
            if action is not None and isinstance(action, (list, tuple)) and action[0] == 5:
                iou = calculate_best_iou([current_state], target_bboxes) if target_bboxes else 0.0
                R_conf = 1.0 if iou >= self.threshold else -1.0

            # Tính R_redundant
            R_redundant = 0.0
            if self.detected_centers:
                for dcx, dcy in self.detected_centers:
                    if np.hypot(cx_new - dcx, cy_new - dcy) < self.alpha * min(self.width, self.height):
                        R_redundant = -1.0
                    else:
                        R_redundant = 1.0

            # Tính R_done
            iou = calculate_best_iou([current_state], target_bboxes) if target_bboxes else 0.0
            dis_to_target = min(np.hypot(cx_new - (b[0] + b[2]) / 2, cy_new - (b[1] + b[3]) / 2) for b in target_bboxes) if target_bboxes else float('inf')
            threshold_distance = self.alpha * min(self.width, self.height)
            if action is not None and isinstance(action, (list, tuple)) and action[0] == 4:
                R_done = min(self.nu * 2 * iou, 1.0) if iou >= self.threshold else -1.0
            else:
                R_done = 0.5 if iou >= self.threshold or dis_to_target < threshold_distance else -0.5

            return 0.4 * R_center + 0.3 * R_class + 0.1 * R_conf + 0.1 * R_redundant + 0.1 * R_done

        else:
            # Tính R_IoU
            iou_current = calculate_best_iou([current_state], target_bboxes) if target_bboxes else 0.0
            iou_previous = calculate_best_iou([previous_state], target_bboxes) if target_bboxes else 0.0
            R_IoU = 1.0 if iou_current > iou_previous else -1.0

            # Tính R_size
            R_size = R_IoU

            # Tính R_conf
            R_conf = 0.0
            if action is not None and action == 5:
                iou = calculate_best_iou([current_state], target_bboxes) if target_bboxes else 0.0
                R_conf = 1.0 if iou >= self.threshold else -1.0

            # Tính R_aspect
            R_aspect = 0.0
            current_aspect = (current_state[2] - current_state[0]) / (current_state[3] - current_state[1]) if (current_state[3] - current_state[1]) > 0 else 1.0
            target_aspect = (target_bboxes[0][2] - target_bboxes[0][0]) / (target_bboxes[0][3] - target_bboxes[0][1]) if (target_bboxes[0][3] - target_bboxes[0][1]) > 0 else 1.0
            R_aspect = 1.0 if abs(current_aspect - target_aspect) < 0.1 else -1.0

            return 0.5 * R_IoU + 0.2 * R_size + 0.1 * R_aspect + 0.2 * R_conf

    def calculate_trigger_reward(self, current_state, target_bboxes):
        iou = calculate_best_iou([current_state], target_bboxes) if target_bboxes else 0.0
        R_done = min(self.nu * 2 * iou, 1.0) if iou >= self.threshold else -1.0
        return R_done

    def transform_action(self, action, phase="center"):
        self.phase = phase
        x1, y1, x2, y2 = self.bbox
        w = x2 - x1
        h = y2 - y1
        dx = int(self.alpha * w)
        dy = int(self.alpha * h)

        if self.phase == "center":
            if isinstance(action, (list, tuple)):
                pos_action = action[0]
            else:
                pos_action = action
            if pos_action == 0:
                x1 += dx
                x2 += dx
            elif pos_action == 1:
                x1 -= dx
                x2 -= dx
            elif pos_action == 2:
                y1 -= dy
                y2 -= dy
            elif pos_action == 3:
                y1 += dy
                y2 += dy
        else:
            if action == 0:
                x1 -= dx
                x2 += dx
                y1 -= dy
                y2 += dy
            elif action == 1:
                x1 += dx
                x2 -= dx
                y1 += dy
                y2 -= dy
            elif action == 2:
                y1 += dy
                y2 -= dy
            elif action == 3:
                x1 += dx
                x2 -= dx

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)
        self.bbox = [x1, y1, x2, y2]
        return self.bbox

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        if self.use_dataset:
            self.extract()
        else:
            self.image = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            self.current_gt_bboxes = [[100, 100, 200, 200]]
            self.current_gt_labels = ["class_0"]

        self.height, self.width = self.image.shape[:2] if self.image is not None else self.target_size
        self.original_image = self.image.copy() if self.image is not None else None
        self.current_gt_index = np.random.randint(0, len(self.current_gt_bboxes)) if self.current_gt_bboxes else 0
        self.target_bbox = self.current_gt_bboxes[self.current_gt_index] if self.current_gt_bboxes else [0, 0, 0, 0]

        start_boxes = [
            [0, 0, int(self.width * 0.75), int(self.height * 0.75)],
            [int(self.width * 0.25), 0, self.width, int(self.height * 0.75)],
            [0, int(self.height * 0.25), int(self.width * 0.75), self.height],
            [int(self.width * 0.25), int(self.height * 0.25), self.width, self.height]
        ]
        ious = [calculate_best_iou([bbox], [self.target_bbox]) for bbox in start_boxes]
        self.bbox = start_boxes[np.argmax(ious)]

        self.phase = "center"
        self.step_count = 0
        self.trigger_count = 0
        self.no_of_triggers = 0
        self.cumulative_reward = 0
        self.done = False
        self.truncated = False
        self.current_action = None
        # self.current_class = None
        self.color = self.generate_random_color()
        self.action_history_center = [[0] * (self.number_actions + self.n_classes) for _ in range(self.action_history_size)]
        self.action_history_size_list = [[0] * self.number_actions for _ in range(self.action_history_size)] # [change]: self.action_history_size -> self.action_history_size_list
        self.classification_dictionary = {"label": [], "confidence": [], "bbox": [], "color": [], "aspect": []}
        self.detected_objects = set()
        self.detected_centers = []

        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()

        self._update_spaces()
        return self.get_state(), self.get_info()

    def get_state(self, dtype=torch.float32):
        x1, y1, x2, y2 = map(int, self.bbox)
        image = self.image.copy() if self.image is not None else np.zeros(self.target_size + (3,), dtype=np.uint8)
        cropped_image = image[y1:y2, x1:x2]

        if cropped_image.size == 0:
            cropped_image = image

        image_tensor = transform_input(cropped_image, target_size=self.target_size)
        with torch.no_grad():
            features = self.feature_extractor(image_tensor.to(self.device))
        features = features.cpu().numpy().flatten()
        features /= np.max(features) if np.max(features) != 0 else 1.0

        # [change]: move center out of self.phrase == center to add centers to size state
        centers = [
                (x1 + x2) / 2 / self.width,
                (y1 + y2) / 2 / self.height
            ]
        centers = np.array(centers, dtype=dtype)
            
        if self.phase == "center":
            
            action_history = np.array([item for sublist in self.action_history_center for item in sublist], dtype=dtype)
            state = np.concatenate([features, centers, action_history])
        else: # Size phase
            action_history = np.array([item for sublist in self.action_history_size_list for item in sublist], dtype=dtype) # [change]: self.action_history_size -> self.action_history_size_list
            state = np.concatenate([features, centers, action_history]) # [change]: add center to size state

        return state.reshape(1, -1)

    def extract(self):
        if not self.dataset or self.current_class not in self.dataset:
            raise ValueError(f"Invalid dataset or current_class {self.current_class}")
        extracted_imgs = self.dataset[self.current_class]
        
        
        if self.class_image_index >= len(extracted_imgs):
            self.class_image_index = 0
            self.epochs += 1
            print(f"\033[92mEpoch {self.epochs} done for class {self.current_class}.\033[0m")
        img_name = list(extracted_imgs.keys())[self.class_image_index]
        img_info = extracted_imgs[img_name][0]
        
        self.image = np.array(img_info[0])
        self.original_image = self.image.copy()
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        gt_bboxes_dict = [item for sublist in img_info[1:] for item in sublist]
        gt_bboxes_dict = [gt_bboxes_dict[i] for i in range(len(gt_bboxes_dict)) if i % 2 == 0]
        self.current_gt_bboxes = [[int(b['xmin']), int(b['ymin']), int(b['xmax']), int(b['ymax'])] for b in gt_bboxes_dict]
        self.current_gt_labels = [b.get('label', f"class_{i}") for i, b in enumerate(gt_bboxes_dict)]
        self.class_names = self.current_class  # Sử dụng danh sách lớp từ env_config['current_class']
        self.target_bbox = self.current_gt_bboxes[0] if self.current_gt_bboxes else [0, 0, 0, 0]
        self.class_image_index += 1
        
        if self.env_mode == 1:
            self.evaluation_results['gt_boxes'][img_name] = self.current_gt_bboxes
            self.evaluation_results['gt_labels'][img_name] = self.current_gt_labels

    def get_info(self):
        return {
            "target_bbox": self.target_bbox,
            "height": self.height,
            "width": self.width,
            "phase": self.phase,
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "iou": calculate_best_iou([self.bbox], self.current_gt_bboxes) if self.current_gt_bboxes else 0.0,
            "recall": calculate_best_recall([self.bbox], self.current_gt_bboxes) if self.current_gt_bboxes else 0.0,
            "gt_bboxes": self.current_gt_bboxes,
            "gt_labels": self.current_gt_labels,
            "classification_dictionary": self.classification_dictionary,
            "env_mode": self.env_mode,
            "epochs": self.epochs,
            "current_class": self.current_class,
        }

    def step(self, action):
        self.current_action = action
        previous_state = self.bbox.copy()
        reward = 0.0

        if self.phase == "center":
            if isinstance(action, (list, tuple)):
                pos_action, class_action = action
            else:
                pos_action, class_action = action, None

            self._update_history([pos_action, class_action] if class_action is not None else [pos_action])

            if pos_action < 4:
                self.bbox = self.transform_action([pos_action, class_action] if class_action is not None else pos_action)
                reward = self.calculate_reward(self.bbox, previous_state, self.current_gt_bboxes, self.current_gt_labels, phase="center", action=[pos_action, class_action] if class_action is not None else pos_action)
                self.step_count += 1
            elif pos_action == 4:
                if self.step_count < self.trigger_steps:
                    reward = -1.0
                else:
                    reward = self.calculate_reward(self.bbox, previous_state, self.current_gt_bboxes, self.current_gt_labels, phase="center", action=[pos_action, class_action] if class_action is not None else pos_action)
                    if reward > 0:
                        self.phase = "size"
                        self._update_spaces()
                        self.step_count = 0
                        self.trigger_count += 1
                        cx, cy = (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2
                        self.detected_centers.append((cx, cy))
            elif pos_action == 5:
                reward = self.calculate_reward(self.bbox, previous_state, self.current_gt_bboxes, self.current_gt_labels, phase="center", action=[pos_action, class_action] if class_action is not None else pos_action)
                self.classification_dictionary["confidence"].append(1.0)
                self.step_count += 1

            if class_action is not None:
                self.current_class = self.class_names[class_action]
                self.classification_dictionary["label"].append(self.current_class)

        else:
            self._update_history(action)
            if action < 4:
                self.bbox = self.transform_action(action)
                reward = self.calculate_reward(self.bbox, previous_state, self.current_gt_bboxes, self.current_gt_labels, phase="size", action=action)
                self.step_count += 1
            elif action == 4:
                if self.step_count < self.trigger_steps:
                    reward = -1.0
                else:
                    reward = self.calculate_trigger_reward(self.bbox, self.current_gt_bboxes)
                    if reward > 0:
                        self.restart_and_change_state()
            else:
                reward = self.calculate_reward(self.bbox, previous_state, self.current_gt_bboxes, self.current_gt_labels, phase="size", action=action)
                self.classification_dictionary["confidence"].append(1.0)
                self.step_count += 1

        self.cumulative_reward += reward

        if self.step_count >= self.max_steps or self.no_of_triggers >= len(self.current_gt_bboxes):
            self.truncated = True
            self.done = True

        if self.render_mode == "human":
            self.render()

        return self.get_state(), reward, self.done, self.truncated, self.get_info()

    def _update_history(self, action):
        if self.phase == "center":
            if isinstance(action, (list, tuple)):
                pos_action, class_action = action
                one_hot = [0] * (self.number_actions + self.n_classes)
                if pos_action < self.number_actions:
                    one_hot[pos_action] = 1
                if class_action is not None:
                    one_hot[self.number_actions + class_action] = 1
                self.action_history_center.pop(0)
                self.action_history_center.append(one_hot)
            else:
                one_hot = [0] * (self.number_actions + self.n_classes)
                if action < self.number_actions:
                    one_hot[action] = 1
                self.action_history_center.pop(0)
                self.action_history_center.append(one_hot)
        else:
            one_hot = [0] * self.number_actions
            one_hot[action] = 1
            self.action_history_size_list.pop(0) # [change]: self.action_history_size -> self.action_history_size_list
            self.action_history_size_list.append(one_hot) # [change]: self.action_history_size -> self.action_history_size_list

    def render(self, mode="rgb_array"):
        if self.image is None:
            return np.zeros(self.target_size + (3,), dtype=np.uint8)

        img = self.original_image.copy() if self.original_image is not None else np.zeros(self.target_size + (3,), dtype=np.uint8)
        x1, y1, x2, y2 = map(int, self.bbox)

        if self.env_mode == 0:
            for idx, bbox in enumerate(self.current_gt_bboxes):
                if idx not in self.detected_objects:
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), self.color, 2)

        if mode == "human":
            image_surface = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
            font = pygame.font.SysFont('Lato', 50)
            action_str = self.decode_render_action(self.current_action) if self.current_action is not None else "N/A"
            text = font.render(f'Phase: {self.phase} | Action: {action_str}', True, (255, 255, 255))
            image_surface.blit(text, (0, 0))
            font_size = int(0.04 * self.height)
            font = pygame.font.SysFont('Lato', font_size)
            text = font.render(
                f'Step: {self.step_count} | Reward: {round(self.cumulative_reward, 3)} | IoU: {round(calculate_best_iou([self.bbox], self.current_gt_bboxes), 3)}',
                True, (255, 255, 255)
            )
            image_surface.blit(text, (0, self.window_size[1] - font_size))
            self.window.blit(image_surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        elif mode == "rgb_array":
            return img

    def display(self, mode):
        img = self.original_image.copy() if self.original_image is not None else np.zeros(self.target_size + (3,), dtype=np.uint8)
        x1, y1, x2, y2 = map(int, self.bbox)

        if mode == "trigger_image":
            cv2.rectangle(img, (x1, y1), (x2, y2), self.color, 2)
        elif mode == "detection":
            for bbox, idx in zip(self.classification_dictionary["bbox"], range(len(self.classification_dictionary["bbox"]))):
                bx1, by1, bx2, by2 = map(int, bbox)
                color = self.classification_dictionary["color"][idx]
                cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
            for gt_bbox in self.current_gt_bboxes:
                gx1, gy1, gx2, gy2 = map(int, gt_bbox)
                cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
        else:
            raise ValueError(f"Mode {mode} not supported!")

        return img

    def restart_and_change_state(self):
        self.image = self.draw_ior_cross(self.image.copy(), self.bbox)
        self.classification_dictionary["bbox"].append(self.bbox)
        self.classification_dictionary["color"].append(self.color)
        self.classification_dictionary["aspect"].append((self.bbox[2] - self.bbox[0]) / (self.bbox[3] - self.bbox[1]))
        if self.current_class is not None:
            self.classification_dictionary["label"].append(self.current_class)
        self.classification_dictionary["confidence"].append(1.0)
        self.no_of_triggers += 1
        self.detected_objects.add(self.current_gt_index)

        remaining_objects = [i for i in range(len(self.current_gt_bboxes)) if i not in self.detected_objects]
        if not remaining_objects:
            self.done = True
            self.current_gt_index = 0
        else:
            self.current_gt_index = random.choice(remaining_objects)
            self.target_bbox = self.current_gt_bboxes[self.current_gt_index]

        self.phase = "center"
        self._update_spaces()
        self.step_count = 0
        self.color = self.generate_random_color()
        self.current_class = None

        start_boxes = [
            [0, 0, int(self.width * 0.75), int(self.height * 0.75)],
            [int(self.width * 0.25), 0, self.width, int(self.height * 0.75)],
            [0, int(self.height * 0.25), int(self.width * 0.75), self.height],
            [int(self.width * 0.25), int(self.height * 0.25), self.width, self.height]
        ]
        ious = [calculate_best_iou([bbox], [self.target_bbox]) for bbox in start_boxes]
        self.bbox = start_boxes[np.argmax(ious)]

    def decode_render_action(self, action):
        if action is None:
            return "N/A"
        if self.phase == "center":
            if isinstance(action, (list, tuple)):
                pos_action, class_action = action
                pos_str = ["Move right", "Move left", "Move up", "Move down"][pos_action] if pos_action < 4 else "Trigger" if pos_action == 4 else "Confidence"
                class_str = f"Class: {self.class_names[class_action]}" if class_action is not None else "No Class"
                return f"{pos_str}, {class_str}"
            else:
                pos_str = ["Move right", "Move left", "Move up", "Move down"][action] if action < 4 else "Trigger" if action == 4 else "Confidence"
                return f"{pos_str}, No Class"
        else:
            if action < 4:
                return ["Make bigger", "Make smaller", "Make fatter", "Make taller"][action]
            elif action == 4:
                return "Trigger"
            else:
                return "Confidence"

    def draw_ior_cross(self, image, bbox):
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        length = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 4
        cv2.line(image, (cx - length, cy - length), (cx + length, cy + length), (255, 0, 0), 2)
        cv2.line(image, (cx - length, cy + length), (cx + length, cy - length), (255, 0, 0), 2)
        return image

    def generate_random_color(self):
        h = random.random()
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
        cv2.destroyAllWindows()
