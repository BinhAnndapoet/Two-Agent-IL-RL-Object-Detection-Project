from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
import cv2


# Constants
NUMBER_OF_ACTIONS = 5       # 4 ops + trigger
ALPHA = 0.2                 # movement/resize scale
NU = 3.0                    # trigger reward factor
THRESHOLD = 0.6             # IoU threshold for trigger
MAX_STEPS = 200             # max steps per episode
TRIGGER_STEPS = 40          # steps before trigger allowed
ACTION_HISTORY_SIZE = 10    # number of past actions to remember
SINGLE_OBJ = 0
MULTI_OBJ = 1
OBJ_CONFIGURATION = SINGLE_OBJ   # default single-object
RENDER_MODE = None          # default render mode

# box format: [x1, y1, x2, y2]
def calculate_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.

    Args:
        box1 (list[int]): [x1, y1, x2, y2] first box coordinates.
        box2 (list[int]): [x1, y1, x2, y2] second box coordinates.

    Returns:
        float: IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

class DetectionEnv(Env):
    """
        Gym environment for two-agent RL object detection.
        Agent behavior switches based on `agent_type` ('center' or 'size').

        State:
            - Normalized bbox coords [x1/w, y1/h, x2/w, y2/h]
            - Flattened one-hot action history (NUMBER_OF_ACTIONS x ACTION_HISTORY_SIZE)

        Actions (Discrete(NUMBER_OF_ACTIONS)):
            0: move left/resize width+
            1: move right/resize width-
            2: move up/resize height-
            3: move down/resize height+
            4: trigger (end)
    """
    
    metadata = {
        'render_modes' : ['human', 'rgb_array'],
        'render_fps' : 5        
    }
    
    def __init__(self, env_config={}):
        """
            Initialize the environment.

            Args:
                env_config (dict): configuration dict with keys:
                    - 'agent_type' (str): 'center' or 'size'
                    - 'image' (np.ndarray): input image array
                    - 'target_bbox' (list of int): ground-truth box [x1,y1,x2,y2]
                    - 'alpha' (float, optional): movement/resize scale
                    - 'nu' (float, optional): trigger reward factor
                    - 'threshold' (float, optional): IoU threshold
                    - 'max_steps' (int, optional): max steps per episode
                    - 'trigger_steps' (int, optional): min steps before trigger
                    - 'obj_configuration' (int, optional): SINGLE_OBJ or MULTI_OBJ
        """
        super().__init__()
        # agent type: 'center' or 'size'
        self.agent_type = env_config.pop('agent_type', None)
        
        # requires input
        self.image = env_config.pop('image', None)
        self.gt_box = env_config.pop("target_bbox", None)
        self.original_image = self.image.copy()
        
        # hyperparameters
        self.alpha      = env_config.pop('alpha', ALPHA)
        self.nu         = env_config.pop('nu', NU)
        self.threshold  = env_config.pop('threshold', THRESHOLD)
        self.max_steps  = env_config.pop('max_steps', MAX_STEPS)
        self.trigger_steps = env_config.pop('trigger_steps', TRIGGER_STEPS)
        self.obj_configuration = env_config.pop('obj_configuration', OBJ_CONFIGURATION)
        
        # image dimension
        h, w = self.image.shape[:2]
        self.width = w
        self.height = h
        
        # action history buffer init
        self.action_history = [[0]*NUMBER_OF_ACTIONS for _ in range(ACTION_HISTORY_SIZE)]
        
        # init episode
        self.reset()
        
        # action & observation spaces
        self.action_space = spaces.Discrete(NUMBER_OF_ACTIONS)
        obs_size = 4 + NUMBER_OF_ACTIONS * ACTION_HISTORY_SIZE
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_size,), dtype=np.float32)

    def reset(self,  *, seed=None, options=None):
        """
            Reset environment to initial state.

            Returns:
                obs (np.ndarray): initial observation
                info (dict): empty info dict
        """
        
        # full-image bbox
        self.bbox = [0, 0, self.width, self.height]
        self.step_count = 0
        self.trigger_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        
        # clear history
        self.action_history = [[0] * NUMBER_OF_ACTIONS for _ in range(ACTION_HISTORY_SIZE)]
        return self.get_state(), {}
    
    def get_state(self):
        """
            Construct state vector from current bbox and action history.

            Returns:
                np.ndarray: state of shape (4 + N_actions * history_size,)
        """
        x1, y1, x2, y2 = self.bbox
        norm = [x1/self.width, y1/self.height, x2/self.width, y2/self.height] # chuẩn hóa về 0, 1
        hist = [bit for vec in self.action_history for bit in vec]
        return np.array(norm + hist, dtype=np.float32)
    
    def update_history(self, action):
        """
            Append last action to history buffer (one-hot), pop oldest.

            Args:
                action (int): action index in [0, NUMBER_OF_ACTIONS-1]
        """
        self.action_history.pop(0)
        onehot = [0]*NUMBER_OF_ACTIONS
        onehot[action] = 1
        self.action_history.append(onehot)
    
    def transform_action(self, action):
        """
            Apply movement or resize to self.bbox based on action.

            Args:
                action (int): in {0,1,2,3}
            Returns:
                list of int: new bbox [x1,y1,x2,y2]
        """
        x1,y1,x2,y2 = self.bbox
        w,h = x2-x1, y2-y1
        dx,dy = int(self.alpha*w), int(self.alpha*h)
        
        # center agent
        if self.agent_type=='center':
            if action==0: 
                x1-=dx
                x2-=dx
            elif action==1: 
                x1+=dx
                x2+=dx
            elif action==2: 
                y1-=dy
                y2-=dy
            elif action==3: 
                y1+=dy
                y2+=dy
        
        # size agent
        else:  
            if action==0: 
                x1-=dx
                x2+=dx
            elif action==1: 
                x1+=dx
                x2-=dx
            elif action==2:
                y1-=dy
                y2+=dy
            elif action==3:
                y1+=dy
                y2-=dy
        # clamp
        return [
            max(0,x1), 
            max(0,y1), 
            min(self.width,x2), 
            min(self.height,y2)
        ]


    def calculate_reward(self, prev_bbox: list, curr_bbox: list = None, target_bbox: list = None, reward_fn = calculate_iou):
        """
            Reward for a non-trigger action:
                +1 if IoU(curr, GT) > IoU(prev, GT)
                -1 otherwise

            Args:
                prev_bbox   (list[int]): bbox before action
                curr_bbox   (list[int], optional): bbox after action (defaults to self.bbox)
                target_bbox (list[int], optional): ground-truth box (defaults to self.gt_box)
                reward_fn   (callable, optional): function to compute IoU

            Returns:
                float: +1.0 or -1.0
        """
        
        if curr_bbox is None:
            curr_bbox = self.bbox
        if target_bbox is None:
            target_bbox = self.gt_box

        iou_prev = reward_fn(prev_bbox, target_bbox)
        iou_curr = reward_fn(curr_bbox, target_bbox)
        return 1.0 if iou_curr > iou_prev else -1.0
        
    
    def calculate_trigger_reward(self, curr_bbox: list = None, target_bbox: list = None, reward_fn = calculate_iou):
        
        """
            Reward for trigger action: if IoU>=threshold, +2*nu*iou, else -nu.

            Returns:
                float: reward_value
        """
        iou = calculate_iou(self.bbox, self.gt_box)
        return (self.nu*2*iou) if iou>=self.threshold else -self.nu

    def step(self, action):
        """
            Take one step in the environment.

            Args:
                action (int): chosen action from action_space
            Returns:
                obs, reward, done, truncated, info
        """
        self.update_history(action)

        if action == NUMBER_OF_ACTIONS-1:  # trigger
            if self.step_count < self.trigger_steps:
                reward = -1.0
            else:
                reward = self.calculate_trigger_reward()
                self.trigger_count += 1
                if self.obj_configuration==SINGLE_OBJ:
                    self.done = True
        else:
            prev = self.bbox.copy()
            self.bbox     = self.transform_action(action)
            reward        = self.calculate_reward(prev)
            self.step_count += 1
            if self.step_count >= self.max_steps:
                self.done = True

        self.cumulative_reward += reward
        return self.get_state(), reward, self.done, False, {}
    
    def render(self, mode='human'):
        """
        Render current bbox on the original image.

        Args:
            mode (str): 'human' to show window, 'rgb_array' to return image
        Returns:
            np.ndarray: rendered frame
        """
        img = self.original_image.copy()
        x1,y1,x2,y2 = self.bbox
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        if mode=='human':
            cv2.imshow('CustomEnv', img)
            cv2.waitKey(1)
        return img

    def close(self):
        """
        Close any opened render windows.
        """
        cv2.destroyAllWindows()
    
    def get_actions(self):
        """
            Function that prints the name of the actions.
        """
        
        pass

    def decode_action(self):
        """
            Function that decodes the action.

            Input:
                - Action to decode

            Output:
                - Unique print of the action
        """
        
        pass

    def rewrap(self):
        """
            Function that rewrap the coordinate if it is out of the image.

            Input:
                - Coordinate to rewrap
                - Size of the image

            Output:
                - Rewrapped coordinate
        """
        pass
    
    def get_info(self):
        """
            Function that returns the information of the environment.

            Output:
                - Information dictionary of the environment
        """
        pass
    
    def generate_random_color(self):
        """
            Function that generates a random color.

            Input:
                - Threshold

            Output:
                - Random color
        """
        pass
    
    
    
    def get_labels(self):
        """
            Function that returns the labels of the images.

            Output:
                - Labels of the images
        """
        pass
    
    def predict(self):
        """
            Function that predicts the label of the image.

            Args:
                - do_display: Whether to display the image or not
                - do_save: Whether to save the image or not
                - save_path: Path to save the image

            Output:
                - Image
        """
        pass

    def restart_and_change_state(self):
        """
            Function that restarts the environment and changes the state.
        """
        pass

    
    
    
    def decode_render_action(self):
        """
        Function that decodes the action.

        Input:
            - Action to decode

        Output:
            - Decoded action as a string
        """
        pass

    def _render_frame(self):
        """
            Function that renders the environment.

            Args:
                - Mode: Mode of rendering (human, trigger_image, bbox, rgb_array)
                - Close: Whether to close the environment or not
                - Alpha: Alpha value for blending the image with the rectangle
                - Text_display: Whether to display the text or not

            Output:
                - Image
        """
        pass

    
    def display(self):
        """
            Function that renders the environment.

            Args:
                - Mode: Mode of rendering (image, trigger_image, bbox, detection, heatmap, None)
                - Do_display: Whether to display the image or not
                - Text_display: Whether to display the text or not
                - Alpha: Alpha value for blending the image with the rectangle
                - Color: Color of the bounding box

            Output:
                - Image of the environment
        """
        pass
    

    