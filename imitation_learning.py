import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from models import ILModel
from env import DetectionEnv
from utils import Transition, device

def select_expert_action_center(env, current_bbox, target_bbox, target_label):
    """
    Select optimal action for CenterDQNAgent based on ground truth.

    Args:
        env (DetectionEnv): Environment instance.
        current_bbox (list): Current bounding box [x1, y1, x2, y2].
        target_bbox (list): Ground truth bounding box [x1, y1, y2, y2].
        target_label (str): Ground truth class label.

    Returns:
        tuple: (pos_action, class_action, conf, done).
            - pos_action (int): 0: right, 1: left, 2: up, 3: down.
            - class_action (int): Class index (0-19).
            - conf (float): Confidence score (0-1).
            - done (float): Done signal (0 or 1).
    """
    current_center = [(current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2]
    target_center = [(target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2]
    dx = target_center[0] - current_center[0]
    dy = target_center[1] - current_center[1]
    alpha_w = env.alpha * (current_bbox[2] - current_bbox[0])
    alpha_h = env.alpha * (current_bbox[3] - current_bbox[1])
    
    # Confidence and done signals
    conf = 1.0  # Assume expert is confident
    done = 1.0 if abs(dx) < alpha_w / 2 and abs(dy) < alpha_h / 2 else 0.0
    
    # Class action
    class_action = env.get_class_names().index(target_label) if target_label in env.get_class_names() else 0
    
    # Position action
    if done:
        pos_action = 4  # Trigger
    else:
        actions = [
            (0, dx > alpha_w),  # Right
            (1, dx < -alpha_w),  # Left
            (2, dy < -alpha_h),  # Up
            (3, dy > alpha_h)   # Down
        ]
        valid_actions = [a[0] for a in actions if a[1]]
        pos_action = random.choice(valid_actions) if valid_actions else 4
    
    return pos_action, class_action, conf, done

def select_expert_action_size(env, current_bbox, target_bbox):
    """
    Select optimal action for SizeDQNAgent based on ground truth.

    Args:
        env (DetectionEnv): Environment instance.
        current_bbox (list): Current bounding box [x1, y1, x2, y2].
        target_bbox (list): Ground truth bounding box [x1, y1, x2, y2].

    Returns:
        tuple: (size_action, conf).
            - size_action (int): 0: bigger, 1: smaller, 2: fatter, 3: taller.
            - conf (float): Confidence score (0-1).
    """
    current_width = current_bbox[2] - current_bbox[0]
    current_height = current_bbox[3] - current_bbox[1]
    target_width = target_bbox[2] - target_bbox[0]
    target_height = target_bbox[3] - target_bbox[1]
    alpha_w = env.alpha * current_width
    alpha_h = env.alpha * current_height
    
    # Confidence
    conf = 1.0  # Assume expert is confident
    
    # Size action
    if (abs(current_width - target_width) < alpha_w and 
        abs(current_height - target_height) < alpha_h):
        size_action = 4  # Trigger
    else:
        actions = [
            (0, current_width < target_width and current_height < target_height),  # Bigger
            (1, current_width > target_width and current_height > target_height),  # Smaller
            (2, current_height > target_height),  # Fatter
            (3, current_width > target_width)     # Taller
        ]
        valid_actions = [a[0] for a in actions if a[1]]
        size_action = random.choice(valid_actions) if valid_actions else 4
    
    return size_action, conf

def generate_expert_trajectory(center_agent, size_agent, env, num_trajectories=1000):
    """
    Generate expert trajectories using ground truth for IL.

    Args:
        center_agent (CenterDQNAgent): Agent for center phase.
        size_agent (SizeDQNAgent): Agent for size phase.
        env (DetectionEnv): Environment instance.
        num_trajectories (int): Number of trajectories to generate.

    Returns:
        dict: Trajectories {'center': [(state, pos_action, class_action, conf, done), ...],
                           'size': [(state, size_action, conf), ...]}.
    """
    trajectories = {'center': [], 'size': []}
    for _ in range(num_trajectories):
        obs, _ = env.reset()
        target_bbox = env.current_gt_bboxes[env.current_gt_index]
        target_label = env.current_gt_labels[env.current_gt_index]
        
        while True:
            if env.phase == 'center':
                pos_action, class_action, conf, done = select_expert_action_center(
                    env, env.bbox, target_bbox, target_label
                )
                action = pos_action if pos_action < 4 else (4 if done > 0.5 else 5 if conf > 0.5 else class_action + 6)
                new_obs, reward, terminated, truncated, info = env.step(action)
                done_flag = terminated or truncated
                trajectories['center'].append((obs, pos_action, class_action, conf, done))
                obs = new_obs
                if info["phase"] == 'size' or done_flag:
                    if info["phase"] == 'size':
                        env.step(4)  # Trigger to switch phase
            else:
                size_action, conf = select_expert_action_size(env, env.bbox, target_bbox)
                action = size_action if size_action < 4 else 5 if conf > 0.5 else 4
                new_obs, reward, terminated, truncated, info = env.step(action)
                done_flag = terminated or truncated
                trajectories['size'].append((obs, size_action, conf))
                obs = new_obs
                if done_flag:
                    break
            if done_flag:
                break
    return trajectories

def train_il_model(env, trajectories, phase='center', epochs=100):
    """
    Train IL model using expert trajectories with composite loss.

    Args:
        env (DetectionEnv): Environment instance.
        trajectories (list): Expert transitions.
        phase (str): Phase ('center' or 'size').
        epochs (int): Number of training epochs.

    Returns:
        ILModel: Trained IL model.
    """
    ninputs = env.get_state().shape[1]
    n_classes = len(env.get_class_names())
    model = ILModel(ninputs, phase, n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    pos_criterion = nn.SmoothL1Loss()
    class_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for t in trajectories:
            state = torch.tensor(t[0], dtype=torch.float32).unsqueeze(0).to(device)
            if phase == 'center':
                pos_action, class_action, conf, done = t[1:]
                pos_pred, class_pred, conf_pred, done_pred = model(state)
                
                L_pos = pos_criterion(pos_pred[0, pos_action], torch.tensor(0.0, device=device)) if pos_action < 4 else torch.tensor(0.0, device=device)
                L_class = class_criterion(class_pred, torch.tensor([class_action], device=device))
                L_conf = bce_criterion(conf_pred, torch.tensor([conf], device=device))
                L_done = bce_criterion(done_pred, torch.tensor([done], device=device))
                loss = 0.4 * L_pos + 0.3 * L_class + 0.15 * L_conf + 0.15 * L_done
            else:
                size_action, conf = t[1:]
                size_pred, conf_pred = model(state)
                
                L_size = pos_criterion(size_pred[0, size_action], torch.tensor(0.0, device=device)) if size_action < 4 else torch.tensor(0.0, device=device)
                L_conf = bce_criterion(conf_pred, torch.tensor([conf], device=device))
                loss = 0.6 * L_size + 0.4 * L_conf
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Phase: {phase}, Loss: {total_loss/len(trajectories):.4f}")
    return model

def initialize_replay_buffer(env, il_model_center, il_model_size, replay_buffer, num_transitions=1000):
    """
    Initialize replay buffer with IL transitions using trained IL models.

    Args:
        env (DetectionEnv): Environment instance.
        il_model_center (ILModel): IL model for center phase.
        il_model_size (ILModel): IL model for size phase.
        replay_buffer (ReplayBuffer): Replay buffer to initialize.
        num_transitions (int): Number of transitions to generate.
    """
    il_model_center.eval()
    il_model_size.eval()
    for _ in range(num_transitions):
        obs, _ = env.reset()
        while True:
            model = il_model_center if env.phase == 'center' else il_model_size
            with torch.no_grad():
                state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                if env.phase == 'center':
                    pos_pred, class_pred, conf_pred, done_pred = model(state)
                    pos_action = torch.argmax(pos_pred, dim=1).item()
                    class_action = torch.argmax(class_pred, dim=1).item() + 6
                    conf_action = 5 if conf_pred.item() > 0 else None
                    done_action = 4 if done_pred.item() > 0 else None
                    action = done_action or conf_action or (pos_action if torch.max(pos_pred).item() > torch.max(class_pred).item() else class_action)
                else:
                    size_pred, conf_pred = model(state)
                    size_action = torch.argmax(size_pred, dim=1).item()
                    conf_action = 5 if conf_pred.item() > 0 else None
                    action = conf_action or size_action
            new_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.append(Transition(obs, action, reward, done, new_obs))
            obs = new_obs
            if done:
                break
