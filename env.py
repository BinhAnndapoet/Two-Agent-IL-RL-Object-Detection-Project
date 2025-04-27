"""
    Custom Gymnasium environments for two-agent sequential object detection using Imitation Learning + Deep Q-Learning.

    Modules:
        - calculate_iou: utility for computing IoU between two bounding boxes.
        - BaseDetectionEnv: abstract base class providing common logic and Gymnasium API (reset, step, rendering, history management).
        - CenterEnv: concrete environment for Center Agent (predicts object center, class, confidence, done).
        - SizeEnv: concrete environment for Size Agent (refines object size and confidence).

    Based on proposal:
        - Two sequential agents (center then size) operate on full-image or patch state.
        - Action spaces and rewards match design: movement/resize, trigger, class/conf/done.
        - State includes normalized bbox coords, history vector, image/patch input.

    Hyperparameters are defined at module-level and collected in HYPERPARAMETERS dict. 
    They can be overridden via the `env_config` passed to constructors; missing keys fall back to defaults.
"""

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
DEFAULT_PATCH_SIZE      = 64        # Size (height and width) of image patch for SizeEnv


# Default sizes of discrete action spaces (from proposal):
"""  
    - For BaseDetectionEnv: Number of base move/resize actions before trigger (common)
    - For CenterEnv: 4 base movement actions + trigger + 10 class prediction + 3 confidence bins + 2 done flags = 24 total
    - For SizeEnv: 4 resize actions + trigger + 3 conf bins + 4 aspect choices + 1 done flag? (simplified to 12)
"""
DEFAULT_CENTER_ACTIONS  = 24        # 4 move + 1 trigger + 10 class + 3 conf + 2 done
DEFAULT_SIZE_ACTIONS    = 12        # 4 resize + 1 trigger + 3 conf + 4 aspect
DEFAULT_BASE_ACTIONS    = 4         # Number of base move/resize actions before trigger


def calculate_iou(bbox1, bbox2):
    """  
        * Args:
            - bbox1 (list[float]) : [x1, y1, x2, y2] - first bbox coordinates
            - bbox2 (list[float]) : [x1, y1, x2, y2] - second bbox coordinates
        * Returns:
            - float: IoU = area(intersection) / area(union), in [0.0, 1.0]
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = max(bbox1[2], bbox2[2])
    y2 = max(bbox1[3], bbox2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = max(0, (bbox1[2] - bbox1[0])) * max(0, (bbox1[3] - bbox1[1]))
    area2 = max(0, (bbox2[2] - bbox2[0])) * max(0, (bbox2[3] - bbox2[1]))
    union = area1 + area2 - inter_area
    
    if union > 0: 
        IoU = (inter_area / union)
    else:
        IoU = 0.0    
    
    return IoU


class BaseDetectionEnv(Env, ABC):
    """  
        * Abstract base env for sequetial object detection. Configured via `env_config` dict.
        
        * Atributes:
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
            
        * Method to override:
            + _init_space()                     : configure action_space and observation_space of each agent
            + _get_state()                      : return current state represation
            + _calc_reward(pred_bbox, curr_bbox): compute reward for non-trigger 
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
        self.patch_size         = env_config.get('patch_size', DEFAULT_PATCH_SIZE)
        
        # Action-space sizes
        self.center_actions     = env_config.get('center_actions', DEFAULT_CENTER_ACTIONS)
        self.size_actions       = env_config.get('size_actions', DEFAULT_SIZE_ACTIONS)
        self.base_actions       = env_config.get('base_actions', DEFAULT_BASE_ACTIONS)
        
        # Internal state
        self.height, self.width = self.image.shape[:2]
        self.action_history     = [[0] * self.base_actions for _ in range(self.history_size)]
        
        # Initialize action/observation spaces and reset
        self._init_spaces()
        self.reset()
    
    
    @abstractmethod
    def _init_spaces(self):
        """Define action_space and observation_space. Must be implemented by subclasses."""
        raise NotImplementedError
    
    
    @abstractmethod
    def _get_state(self):
        """Return current observation: image/patch tensor + history channels."""
        raise NotImplementedError
    
    
    @abstractmethod
    def _calc_reward(self, prev_bbox: list, curr_bbox: list) -> float:
        """Compute reward for non-trigger actions."""
        raise NotImplementedError
        
    
    # After *: keyword-only paramters
    def reset(self,  *, seed=None, options=None):
        """
            Reset environment to initial state.

            * Args:
                + seed (int, optional): random seed for environment.
                + options (dict, optional): additional reset options.

            * Returns:
                + obs (np.ndarray): initial observation state.
                + info (dict): auxiliary information (empty by default).
        """
        
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
        
    
    def _transform_action(self, action:int):
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
                iou = calculate_iou(self.bbox, self.gt_box)
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
            'iou': calculate_iou(self.bbox, self.gt_box),
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
    
    
class CenterEnv(BaseDetectionEnv):
    def _init_spaces(self):
        pass
    
    
    def _get_state(self):
        pass
    
    
    def _calc_reward(self, prev_bbox: list, curr_bbox: list) -> float:
        pass

class SizeEnv(BaseDetectionEnv):
    def _init_spaces(self):
        pass
    
    
    def _get_state(self):
        pass
    
    
    def _calc_reward(self, prev_bbox: list, curr_bbox: list) -> float:
        pass
    