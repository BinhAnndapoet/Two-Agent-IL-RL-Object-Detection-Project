"""
    Custom Gymnasium environments for two-agent sequential object detection using Imitation Learning + Deep Q-Learning.

    Modules:
    - calculate_iou: utility for computing IoU between two bounding boxes.
    - BaseDetectionEnv: abstract base class providing common Gym API (reset, step, render, close) and history management.
    - CenterEnv: environment for Center Agent (adjust center of bbox).
    - SizeEnv: environment for Size Agent (adjust size of bbox).

    Based on proposal:
        - Two sequential agents: center then size, acting on full-image state.
        - Action spaces: base move/resize and trigger.
        - Rewards: IoU improvement for IL expert, trigger reward based on IoU threshold.
        - State: normalized bbox coordinates + history of past base actions.

    Hyperparameters are defined at module-level and collected in HYPERPARAMETERS dict. 
    They can be overridden via the `env_config` passed to constructors; missing keys fall back to defaults.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from utils import calculate_iou, calculate_best_iou, calculate_best_recall, transform_input
from models import ResNet18FeatureExtractor
import torch
import random
import colorsys
import pygame

class DetectionEnv(gym.Env):
    def __init__(self, env_config:dict):
        super().__init__()
        # config variables  
        self.dataset                    = env_config.get("dataset", [])
        self.alpha                      = env_config.get("alpha", 0.1)
        self.nu                         = env_config.get("nu", 0.5)
        self.threshold                  = env_config.get("threshold", 0.6)
        self.max_steps                  = env_config.get("max_steps", 100)
        self.trigger_steps              = env_config.get("trigger_steps", 5)
        self.number_actions             = env_config.get("number_actions", 6) # base: 4 move/aspect + 1 trigger + 1 conf
        self.action_history_size        = env_config.get("action_history_size", 7)  
        self.obj_config                 = env_config.get("object_config", "SINGLE_OBJECT")
        self.n_classes                  = env_config.get("n_classes", 20) # theo VOC2012
        self.target_size                = env_config.get("target_size", (448, 448))
        self.feature_dim               = env_config.get("feature_dim", 4096)
        self.device                     = env_config.get("device", "cpu")
        self.use_dataset                = env_config.get("use_dataset", True)
        self.env_mode                   = env_config.get("env_mode", 0)  # 0: TRAIN_MODE
        self.epochs                     = env_config.get("epochs", 100)
        self.current_class              = env_config.get("current_class", None)
        self.window_size                = env_config.get("window_size", (448, 448))
        self.metadata                   = {"render_fps": 30}
        self.clock                      = pygame.time.Clock()
        
        
        # Biến trạng thái môi trường
        self.phase                      = "center"      
        self.image                      = None
        self.width                      = self.image[1]
        self.height                     = self.image[0]
        self.bbox                       = [0, 0, 0, 0]  
        self.current_gt_bboxes          = []
        self.current_gt_index           = 0
        self.step_count                 = 0             
        self.trigger_count              = 0             
        self.cumulative_reward          = 0
        self.done                       = False
        self.truncated                  = False
        
        # Tách action_history cho từng phase
        self.action_history_center      = [[0] * (self.number_actions + self.n_classes) for _ in range(self.action_history_size)]  # 7 x 26
        self.action_history_size        = [[0] * self.number_actions for _ in range(self.action_history_size)]  # 7 x 6
        self.render_mode                = "rgb_array"   
        self.feature_extractor          = ResNet18FeatureExtractor().to(self.device)
        self.classification_dictionary  = {"label": [], "confidence": [], "bbox": [], "color": [], "aspect": []}
        
        self._update_spaces()
        
    def _update_spaces(self):
        """Cập nhật action_space và observation_space dựa trên phase."""
        # Action space
        if self.phase == "center": 
            self.action_space   = spaces.Discrete(self.number_actions + self.n_classes)  # Center: 4 move + 1 trigger + 1 conf  + n_classes = 4 + 1 + 1 + 20 = 26
        else:
            self.action_space   = spaces.Discrete(self.number_actions)  # Size: 4 aspect + 1 trigger + 1 conf = 4 + 1 + 1 = 6
            

        # Observation space: Đặc trưng ảnh (feature_dim) + lịch sử hành động
        if self.phase == "center":
            history_dim = self.action_history_size * (self.number_actions + self.n_classes)  # 7 * 26 = 182
            state_dim   = self.feature_dim + 2 + history_dim  # # features + centers + history
        else:
            history_dim = self.action_history_size * self.number_actions  # 7 * 6 = 42
            state_dim   = self.feature_dim + history_dim  # features + history
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )

    def get_current_gt_bboxes(self):
        """Trả về danh sách ground truth bboxes."""
        return self.current_gt_bboxes

    def set_current_gt_bboxes(self, value):
        """Cập nhật danh sách ground truth bboxes."""
        self.current_gt_bboxes = value if isinstance(value, list) else [value]
    
    def calculate_reward(self, current_state, previous_state, target_bboxes, phase="center"):
        self.phase = phase
        
        # Calculate reward for center phase based on Euclidean distance to nearest ground truth.
        if self.phase == "center":
            cx_new      = (current_state[0] + current_state[2]) / 2
            cy_new      = (current_state[1] + current_state[3]) / 2
            cx_prev     = (previous_state[0] + previous_state[2]) / 2
            cy_prev     = (previous_state[1] + previous_state[3]) / 2
            
            dis_new     = min(np.hypot(cx_new - (b[0] + b[2]) / 2, cy_new - (b[1] + b[3]) / 2) for b in target_bboxes)
            dis_prev    = min(np.hypot(cx_prev - (b[0] + b[2]) / 2, cy_prev - (b[1] + b[3]) / 2) for b in target_bboxes)
            
            return 1.0 if dis_new < dis_prev else -1.0

        else:
            iou_current = calculate_best_iou([current_state], target_bboxes)
            iou_previous = calculate_best_iou([previous_state], target_bboxes)
            
            return 1.0 if iou_current > iou_previous else -1.0
        
        
    def calculate_trigger_reward(self, current_state, target_bboxes):
        iou = calculate_best_iou([current_state], target_bboxes)
        
        return min(self.nu * 2 * iou, 1.0) if iou >= self.threshold else -1.0
    
    def transform_action(self, action, phase="center"):
        self.phase = phase
        x1, y1, x2, y2  = self.bbox
        w               = x2 - x1
        h               = y2 - y1
        dx              = int(self.alpha * w)
        dy              = int(self.alpha * h)

        if self.phase == "center":
            if action == 0:     # Right
                x1 += dx
                x2 += dx
            elif action == 1:   # Left
                x1 -= dx
                x2 -= dx
            elif action == 2:   # Up
                y1 -= dy
                y2 -= dy
            elif action == 3:   # Down
                y1 += dy
                y2 += dy
            # action 4: Trigger (xử lý trong step)
            # action 5: conf
            # action 6-25: Class prediction (xử lý trong step)

        else:  # size phase
            if action == 0:     # Bigger
                x1 -= dx
                x2 += dx
                y1 -= dy
                y2 += dy
            elif action == 1:  # Smaller
                x1 += dx
                x2 -= dx
                y1 += dy
                y2 -= dy
            elif action == 2:  # Fatter
                y1 += dy
                y2 -= dy
            elif action == 3:  # Taller
                x1 += dx
                x2 -= dx
                
            # action 4: Trigger (xử lý trong step)
            # action 5: Confidence (xử lý trong step)

        # Giới hạn tọa độ
        x1, y1      = max(0, x1), max(0, y1)
        x2, y2      = min(self.width, x2), min(self.height, y2)
        self.bbox   = [x1, y1, x2, y2]
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

        self.height, self.width = self.image.shape[:2]
        self.original_image = self.image.copy()
        self.current_gt_index = 0 if self.obj_config == "SINGLE_OBJECT" else np.random.randint(0, len(self.current_gt_bboxes))
        self.target_bbox = self.current_gt_bboxes[self.current_gt_index] if self.current_gt_bboxes else []
        self.bbox = [0, 0, 0, 0]
        self.phase = "center"
        self.step_count = 0
        self.trigger_count = 0
        self.no_of_triggers = 0
        self.cumulative_reward = 0
        self.done = False
        self.truncated = False
        self.current_action = None
        self.current_class = None
        self.color = self.generate_random_color()
        self.action_history_center      = [[0] * (self.number_actions + self.n_classes) for _ in range(self.action_history_size)]  # 7 x 26
        self.action_history_size        = [[0] * self.number_actions for _ in range(self.action_history_size)]  # 7 x 6
        self.classification_dictionary = {"label": [], "confidence": [], "bbox": [], "color": [], "aspect": []}

        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()

        self._update_spaces()
        return self.get_state(), self.get_info()
    
    
    def get_state(self, dtype=torch.float32):
        # Bước 1: Crop ảnh theo bbox
        x1, y1, x2, y2  = map(int, self.bbox)
        image           = self.image.copy()
        cropped_image   = image[y1:y2, x1:x2]
        
        # if image is not empty
        if cropped_image.size == 0: # use origin image
            image           = self.image.copy()
            self.truncated  = True
        
        # Bước 2: Transform ảnh và trích xuất đặc trưng   
        image_tensor    = transform_input(cropped_image, target_size=self.target_size)
        
        # retrieving the features of the image with the high level
        with torch.no_grad():
            features = self.feature_extractor(image_tensor) # [4096]
        features = features.cpu().numpy()
        features /= np.max(features)
        
        
        if self.phase == "center":
            centers = [
                (x1 + x2) / 2 / self.width,
                (y1 + y2) / 2 / self.height
            ]
            centers = np.array(centers, dtype=dtype)
            action_history = np.array([item for sublist in self.action_history_center for item in sublist], dtype=dtype)
            state = np.concatenate([features, centers, action_history])
        else:
            centers = np.array([], dtype=dtype)
            action_history = np.array([item for sublist in self.action_history_size for item in sublist], dtype=dtype)
            state = np.concatenate([features, action_history])

            
        return state.reshape(1, -1)
    
    def extract(self):
        """
        Extract image and ground truth from dataset.
        """
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
        self.target_bbox = self.current_gt_bboxes[0]
        self.class_image_index += 1
        if self.env_mode == 1:  # TEST_MODE
            self.evaluation_results['gt_boxes'][img_name] = self.current_gt_bboxes
    
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
            self._update_history(action)
            if action < 4:  # Hành động di chuyển
                self.bbox = self.transform_action(action)
                reward = self.calculate_reward(self.bbox, previous_state, self.current_gt_bboxes, phase="center")
                self.step_count += 1
            elif action == 4:  # Trigger
                if self.step_count < self.trigger_steps_center:
                    reward = -1.0
                else:
                    reward = self.calculate_trigger_reward(self.bbox, self.current_gt_bboxes)
                    if reward > 0:  # Chuyển sang phase size nếu trigger thành công
                        self.phase = "size"
                        self._update_spaces()
                        self.step_count = 0
                        self.trigger_count += 1
            # elif action >= 5:  # Dự đoán lớp
            #     reward = 0.0
        else:  # Phase "size"
            self._update_history(action)
            if action < 4:  # Hành động thay đổi kích thước
                self.bbox = self.transform_action(action)
                reward = self.calculate_reward(self.bbox, previous_state, self.current_gt_bboxes, phase="size")
                self.step_count += 1
            elif action == 4:  # Trigger
                if self.step_count < self.trigger_steps_size:
                    reward = -1.0
                else:
                    reward = self.calculate_trigger_reward(self.bbox, self.current_gt_bboxes)
                    if reward > 0:
                        self.restart_and_change_state()
            else:  # Aspect
                reward = 0.0

        self.cumulative_reward += reward

        if self.step_count >= self.max_steps or self.no_of_triggers >= self.trigger_steps:
            self.truncated = True
            self.done = True

        if self.render_mode == "human":
            self.render()

        return self.get_state(), reward, self.done, self.truncated, self.get_info()
    
    def _update_history(self, action_idx):
        # Xác định số chiều của vector one-hot theo phase
        one_hot_size = (self.number_actions + self.n_classes) if self.phase == "center" else self.number_actions
        
        # Tạo vector one-hot cho hành động
        one_hot = [0] * one_hot_size
        one_hot[action_idx] = 1
        
        # Cập nhật lịch sử hành động (FIFO) theo phase
        if self.phase == "center":
            self.action_history_center.pop(0)
            self.action_history_center.append(one_hot)
        else:
            self.action_history_size.pop(0)
            self.action_history_size.append(one_hot)
        
    def render(self, mode="rgb_array"):
        """Hiển thị môi trường."""
        if self.image is None:
            return np.zeros(self.target_size + (3,), dtype=np.uint8)

        img = self.original_image.copy()
        x1, y1, x2, y2 = map(int, self.bbox)

        if self.env_mode == 0:  # TRAIN_MODE
            for bbox in self.current_gt_bboxes:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), self.color, 2)

        if mode == "human":
            image_surface = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
            font = pygame.font.SysFont('Lato', 50)
            text = font.render(f'Phase: {self.phase} | Action: {self.decode_render_action(self.current_action)}', True, (255, 255, 255))
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
        else:
            raise ValueError(f"Chế độ {mode} không được hỗ trợ!")
    
    def display(self, mode):
        """Hiển thị trạng thái môi trường với chế độ cụ thể."""
        img = self.original_image.copy() if self.original_image is not None else np.zeros(self.target_size + (3,), dtype=np.uint8)
        x1, y1, x2, y2 = map(int, self.bbox)

        if mode == "trigger_image":
            cv2.rectangle(img, (x1, y1), (x2, y2), self.color, 2)
        elif mode == "detection":
            cv2.rectangle(img, (x1, y1), (x2, y2), self.color, 2)
            for gt_bbox in self.current_gt_bboxes:
                gx1, gy1, gx2, gy2 = map(int, gt_bbox)
                cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
        else:
            raise ValueError(f"Chế độ {mode} không được hỗ trợ!")

        return img
    
    def restart_and_change_state(self):
        """Khởi động lại và chuyển sang đối tượng tiếp theo."""
        self.image = self.draw_ior_cross(self.image.copy(), self.bbox)
        self.classification_dictionary["bbox"].append(self.bbox)
        self.no_of_triggers += 1
        self.current_gt_index += 1
        if self.current_gt_index >= len(self.current_gt_bboxes):
            self.current_gt_index = 0
            self.done = True
        self.target_bbox = self.current_gt_bboxes[self.current_gt_index] if self.current_gt_bboxes else []
        self.phase = "center"
        self._update_spaces()
        self.step_count = 0

        start_boxes = [
            [0, 0, int(self.width * 0.75), int(self.height * 0.75)],
            [int(self.width * 0.25), 0, self.width, int(self.height * 0.75)],
            [0, int(self.height * 0.25), int(self.width * 0.75), self.height],
            [int(self.width * 0.25), int(self.height * 0.25), self.width, self.height]
        ]
        ious = [calculate_best_iou([bbox], [self.target_bbox]) for bbox in start_boxes]
        self.bbox = start_boxes[np.argmax(ious)]
    
    def decode_render_action(self, action):
        """Giải mã hành động để hiển thị."""
        if action is None:
            return "N/A"
        if self.phase == "center":
            if action < 4:
                return ["Move right", "Move left", "Move up", "Move down"][action]
            elif action == 4:
                return "Trigger"
            elif action == 5:
                return "Confidence"
            else:
                return f"Class: {self.n_classes[action - 5]}"
        else:
            if action < 4:
                return ["Make bigger", "Make smaller", "Make fatter", "Make taller"][action]
            elif action == 4:
                return "Trigger"
            else:
                return "Confidence"

    
    def draw_ior_cross(self, image, bbox):
        """Vẽ dấu chéo tại tâm của bounding box."""
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        length = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 4
        cv2.line(image, (cx - length, cy - length), (cx + length, cy + length), (255, 0, 0), 2)
        cv2.line(image, (cx - length, cy + length), (cx + length, cy - length), (255, 0, 0), 2)
        return image
    
    def generate_random_color(self):
        """Tạo màu ngẫu nhiên cho bounding box."""
        h = random.random()
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def close(self):
        """Đóng môi trường."""
        if self.render_mode == "human":
            pygame.quit()
        cv2.destroyAllWindows()

