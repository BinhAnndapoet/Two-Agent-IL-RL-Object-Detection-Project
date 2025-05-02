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

from utils import iou

from abc import ABC, abstractmethod
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
import cv2

# Default hyperparameters and action-space sizes
DEFAULT_ALPHA           = 0.2       # Fraction of bbox width/height to move/resize per action
DEFAULT_NU              = 3.0       # Scaling factor for trigger reward magnitude
DEFAULT_THRESHOLD       = 0.6       # IoU threshold above which triggering yields positive reward
DEFAULT_MAX_STEPS       = 200       # Max number of non-trigger actions before episode truncation
DEFAULT_TRIGGER_STEPS   = 40        # Minimum steps before trigger action is considered valid
DEFAULT_HISTORY_SIZE    = 10        # Number of past base actions to include in state history
DEFAULT_OBJ_CONFIG      = 0         # Object configuration: 0=SINGLE_OBJ, 1=MULTI_OBJ


# Default sizes of discrete action spaces (from proposal):
"""  
    - For BaseDetectionEnv: Number of base move/resize actions before trigger (common)
    - For CenterEnv: 4 base movement actions + trigger + 10 class prediction + 3 confidence bins + 2 done flags = 24 total
    - For SizeEnv: 4 resize actions + trigger + 3 conf bins + 4 aspect choices + 1 done flag? (simplified to 12)
"""
DEFAULT_CENTER_ACTIONS  = 24        # 4 move + 1 trigger + 10 class + 3 conf + 2 done
DEFAULT_SIZE_ACTIONS    = 12        # 4 resize + 1 trigger + 3 conf + 4 aspect
DEFAULT_BASE_ACTIONS    = 4         # Number of base move/resize actions before trigger


class BaseDetectionEnv(Env, ABC):
    """  
        * Abstract base env for sequetial object detection. Configured via `env_config` dict.
        
        * Attributes:
            + img(np.array)                     : input RGB images
            + gt_bbox(list[float])              : ground_size scale
            + nu(float)                         : trigger reward factor
            + threshold(float)                  : IoU cutoff for trigger reward
            + max_steps (int)                   : maximun steps before valid trigger
            + history_size(int)                 : length of action history.
            + n_base_history(int)               : number of move/resize ops (4).
            + action_history(list[list[int]])   : sliding window of one-hot base actions
            + obj_config(int)                   : SINGLE_OBJ or MULTI_OBJ mode.
            + patch_size(int)                   : dimension for crop in SizeEnv.
            + center_actions(int)               : total actions in CenterEnv.
            + size_actions(int)                 : total actions in SizeEnv.
        
        * Common method:
            + reset()                           : initialize esiode variables and return initial state.
            + step(action)                      : apply action, compute reward, return (obs, reward, done, truncated, info)
            + render(mode)                      : visualize bbox on image
            + close()                           : close render windows
            
        * Abstract -> Method to override:
            + _init_space()                     : configure action_space and observation_space of each agent
            + _get_state()                      : return current state represation
            + calculate_reward()                : compute reward for non-trigger 
    """
    
    
    def __init__(self, env_config:dict):
        super().__init__()
        
        # requires configuration keys
        self.image              = env_config['image']
        self.gt_bbox            = env_config['target_bbox']
        self.agent_type         = env_config['agent_type'] # center or size
        
        # hyperparameters
        self.alpha              = env_config.get('alpha', DEFAULT_ALPHA)
        self.nu                 = env_config.get('nu', DEFAULT_NU)
        self.threshold          = env_config.get('threshold', DEFAULT_THRESHOLD)
        self.max_steps          = env_config.get('max_steps', DEFAULT_MAX_STEPS)
        self.trigger_steps      = env_config.get('trigger_steps', DEFAULT_TRIGGER_STEPS)
        self.history_size       = env_config.get('history_size', DEFAULT_HISTORY_SIZE)
        self.obj_config         = env_config.get('obj_config', DEFAULT_OBJ_CONFIG)
        
        # Action-space sizes
        self.center_actions     = env_config.get('center_actions', DEFAULT_CENTER_ACTIONS)
        self.size_actions       = env_config.get('size_actions', DEFAULT_SIZE_ACTIONS)
        self.base_actions       = env_config.get('base_actions', DEFAULT_BASE_ACTIONS)
        
        # Internal state
        self.height, self.width = self.image.shape[:2]
        self.action_history     = [[0] * self.base_actions for _ in range(self.history_size)]
        self.step_count = 0
        self.trigger_count = 0
        self.done = False
        self.epochs = 0
        
        # Initialize action/observation spaces
        self._init_spaces()
    
    
    @abstractmethod
    def _init_spaces(self):
        """Define action_space and observation_space. Must be implemented by subclasses."""
        raise NotImplementedError
    
    
    @abstractmethod
    def _get_state(self):
        """
        Build and return current raw observation as 1D numpy array.

        Returns:
            np.ndarray: shape (D,), where D = 4 + history_size*base_actions.
        """
        raise NotImplementedError
    
    
    @abstractmethod
    def calculate_reward(self, new_bbox, prev_bbox: list, target_bboxes: list) -> float:
        """
            Abstract reward function used by IL expert and RL.

            Args:
                new_bbox (list[int]): bbox after action.
                prev_bbox (list[int]): bbox before action.
                target_bboxes (list): ground-truth bbox(es).
            Returns:
                float: reward value.
        """
        raise NotImplementedError
    
    
    def get_state(self) -> np.array:
        """
            Return current observation with batch dimension for agent.

            Returns:
                np.ndarray: shape (1, D).
        """
        raw = self._get_state()
        return raw.reshape(1, -1)
        
    
    # After *: keyword-only paramters
    def reset(self,  *, seed=None, options=None):
        """
            Reset environment to initial state for new episode

            * Args:
                + seed (int, optional): random seed for environment.
                + options (dict, optional): additional reset options.

            * Returns:
                + obs (np.ndarray): initial observation state - raw state (no batch dim)
                + info (dict): auxiliary information (empty by default).
        """
        
        self.epochs += 1
        
        # full-image bbox
        self.bbox = [0, 0, self.width, self.height]
        self.step_count = 0
        self.trigger_count = 0
        self.done = False
        
        # clear history
        self.action_history = [[0] * self.base_actions for _ in range(self.history_size)]
        
        return self._get_state(), {}
    
    
    def _update_history(self, action_idx:int):
        """
            Update sliding window of past base actions.
            * Args:
                + action_idx (int): index of the last performed base action.

            * Returns:
                + None
        """
        
        self.action_history.pop(0)          # erase the lastest action in action_history
        one_hot = [0] * self.base_actions   # create new list one hot
        if action_idx < self.base_actions:  # check if action_idx in [0, base_action - 1] 
            one_hot[action_idx] = 1
        self.action_history.append(one_hot)
        
    
    def transform_action(self, action:int):
        """
            Updated coordinates of bbox by applied move or resized to the bounding box based on the action and alpha.

            * Computes step sizes dx, dy as fractions of current bbox width and height:
                    + dx = int(alpha * width)
                    + dy = int(alpha * height)
                    
            * For 'center' agent: moves the bbox by (dx, dy) in one of four directions.
            * For 'size' agent  : expands or contracts the bbox dimensions by dx or dy.
            
            * Args:
                + action (int): index of the action to apply.

            * Returns:
                + new_bbox (list[int]): updated bounding box [x1, y1, x2, y2].
        """
        
        x1, y1, x2, y2 = self.bbox
        w   = x2-x1,
        h   = y2-y1
        dx  = int(self.alpha*w) # horizontal step (pixels)
        dy  = int(self.alpha*h) # vertical   step (pixels)
        
        # center agent (moves the entire bounding box in four directions: left, right, up, down)
        if self.agent_type == 'center':
            if action == 0:       # move left
                x1 -= dx
                x2 -= dx
            elif action == 1:     # move right
                x1 += dx
                x2 += dx
            elif action == 2:     # move up
                y1 -= dy
                y2 -= dy
            elif action == 3:     # move down
                y1 += dy
                y2 += dy
        
        # size agent (expands or contracts the bounding box dimensions horizontally or vertically)
        else:  
            if action == 0:       # expand horizontal dimension 
                x1 -= dx
                x2 += dx
            elif action == 1:     # contract horizontal dimension
                x1 += dx
                x2 -= dx
            elif action == 2:     # expand vertical dimension
                y1 -= dy
                y2 += dy
            elif action == 3:     # contract vertical dimension
                y1 += dy
                y2 -= dy
        
        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)
        
        return [x1, y1, x2, y2]


    def step(self, action: int):
        """
        Execute one environment step given an action.

        Args:
            action (int): index of the action to apply. 
                - 0..(n_base-1): move/resize operations
                - n_base (last index): trigger action

        Returns:
            obs (np.ndarray): next observation state (image/patch + history channels).
            reward (float): scalar reward for the action.
            done (bool): True if episode ended by valid trigger (and single-object mode).
            truncated (bool): True if episode ended by exceeding max_steps.
            info (dict): supplementary info containing:
                - 'iou': current IoU with ground truth box.
                - 'step': total non-trigger steps taken.
                - 'triggered': True if action was the trigger action.
        """
        # step 1: Record this action in history buffer
        self._update_history(action)
        truncated = False

        # step 2: Handle trigger action (last discrete index)
        if action == self.action_space.n - 1:
            # 2.1: If triggered too early, apply penalty
            if self.step_count < self.trigger_steps:
                reward = -1.0
            else:
                # 2.2: Compute IoU and assign trigger reward or penalty
                iou = calculate_iou(self.bbox, self.gt_bbox)
                if iou >= self.threshold:
                    reward = 2 * self.nu * iou
                else:
                    reward = -self.nu
                    
                # Count triggers and end episode if in single-object mode
                self.trigger_count += 1
                if self.obj_config == DEFAULT_OBJ_CONFIG:
                    self.done = True
        else:
            # step 3: Handle move/resize actions
            prev_bbox = list(self.bbox)                        # save current box
            self.bbox = self._transform_action(action)         # update box coords
            reward = self._calc_reward(prev_bbox, self.bbox)   # +1 if IoU improved, else -1
            self.step_count += 1
            
            # End if max steps reached
            if self.step_count >= self.max_steps:
                truncated = True
                self.done = True

        # step 4: Build next observation and info
        obs = self._get_state()
        info = {
            'iou': calculate_iou(self.bbox, self.gt_bbox),
            'step': self.step_count,
            'triggered': (action == self.action_space.n - 1)
        }
        
        # step 5: Return step tuple
        return obs, reward, self.done, truncated, info
        

    def render(self, mode='human'):
        """
            Render bounding box on the original image.

            * Args:
                mode (str): 'human' to display window, 'rgb_array' to return image array.

            * Returns:
                frame (np.ndarray): annotated image frame.
        """
        
        frame = self.image.copy()
        x1, y1, x2, y2 = self.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if mode == 'human':
            cv2.imshow('DetectionEnv', frame)
            cv2.waitKey(1)
        return frame


    def close(self):
        """
            Close any opened render windows.
        """
        cv2.destroyAllWindows()
        
    
    @property
    def current_gt_bboxes(self):
        """
            Return list of ground-truth bboxes.

            Returns:
                list[list[float]]
        """ 
        if isinstance(self.gt_bbox[0], (list, tuple)):
            return self.gt_bbox
        return [self.gt_bbox]
    
    
class CenterEnv(BaseDetectionEnv):
    def _init_spaces(self):
        """
            Initialize the action and observation spaces for the Center agent.

            Action space:
                Discrete(self.center_actions)
            Observation space:
                Box(low=0.0, high=1.0, shape=(D,), dtype=float) where D = 4 + history_size*base_actions

            Returns:
                None
        """
        self.action_space = spaces.Discrete(self.center_actions)
        dim = 4 + self.history_size * self.base_actions
        self.observation_space = spaces.Box(0.0, 1.0, shape=(dim,), dtype=np.float32)
    
    
    def _get_state(self):
        """
            Build and return the current observation for the Center agent.

            The state vector concatenates:
                - normalized bbox coordinates [x1/W, y1/H, x2/W, y2/H]
                - flattened history of past base actions

            Returns:
                np.ndarray: 1D array of length D = 4 + history_size*base_actions
        """
        x1, y1, x2, y2 = self.bbox
        norm = [x1/self.width, y1/self.height, x2/self.width, y2/self.height]
        history = [v for rec in self.action_history for v in rec]
        return np.array(norm + history, dtype=np.float32)
    
    def calculate_reward(self, new_bbox, prev_bbox, target_bboxes):
        """
            Compute reward for a base action in the Center phase.

            Uses Euclidean distance between bbox center and ground truth center:
                +1.0 if distance to GT decreases after action
                -1.0 otherwise

            Args:
                new_bbox (list[int]): bbox after the action
                prev_bbox (list[int]): bbox before the action
                target_bboxes (list[list[int]]): list of ground-truth boxes


            Returns:
                float: reward value based on center distance improvement
        """
        
        # helper to compute center of a bbox
        def _center(b):
            return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        
        cx_prev, cy_prev = _center(prev_bbox)
        cx_new, cy_new = _center(new_bbox)
        cx_gt, cy_gt = _center(target_bboxes[0])
        
        # compute Euclidean distances
        dis_prev = np.hypot(cx_prev - cx_gt, cy_prev - cy_gt)
        dis_new = np.hypot(cx_new - cx_gt, cy_new - cy_gt)
        
        return 1.0 if dis_new < dis_prev else -1.0

class SizeEnv(BaseDetectionEnv):
    def _init_spaces(self):
        """
            Initialize the action and observation spaces for the Size agent.

            Action space:
                Discrete(self.size_actions)
            Observation space:
                Box(low=0.0, high=1.0, shape=(D,), dtype=float) where D = 4 + history_size*base_actions

            Returns:
                None
        """
        
        self.action_space = spaces.Discrete(self.size_actions)
        dim = 4 + self.history_size * self.base_actions
        self.observation_space = spaces.Box(0.0, 1.0, shape=(dim,), dtype=np.float32)
    
    
    def _get_state(self):
        """
            Build and return the current observation for the Size agent.

            The state vector concatenates:
                - normalized bbox coordinates [x1/W, y1/H, x2/W, y2/H]
                - flattened history of past base actions

            Returns:
                np.ndarray: 1D array of length D = 4 + history_size*base_actions
        """
        x1, y1, x2, y2 = self.bbox
        norm = [x1/self.width, y1/self.height, x2/self.width, y2/self.height]
        history = [v for rec in self.action_history for v in rec]
        return np.array(norm + history, dtype=np.float32)
    
    
    def calculate_reward(self, new_bbox, prev_bbox, target_bboxes, phase):
        """
        Compute reward for a base action in the Size phase.

        Uses IoU improvement as metric:
            +1.0 if IoU(new_bbox, GT) > IoU(prev_bbox, GT)
            -1.0 otherwise

        Args:
            new_bbox (list[int]): bbox after the action
            prev_bbox (list[int]): bbox before the action
            target_bboxes (list[list[int]]): list of ground-truth boxes
            phase (str): 'size'

        Returns:
            float: reward value
        """
        prev_iou = calculate_iou(prev_bbox, target_bboxes[0])
        new_iou  = calculate_iou(new_bbox, target_bboxes[0])
        return 1.0 if new_iou > prev_iou else -1.0
    
