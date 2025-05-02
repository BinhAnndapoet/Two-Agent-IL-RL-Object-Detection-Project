import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from env import DetectionEnv
from models import ILModel
from utils import Transition, ReplayBuffer

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

def generate_expert_trajectory(center_agent, size_agent, env, num_trajectories=1000):
    """
    Generate expert trajectories using ground truth for IL.

    Args:
        center_agent (CenterDQNAgent): Agent for center phase.
        size_agent (SizeDQNAgent): Agent for size phase.
        env (DetectionEnv): Environment instance.
        num_trajectories (int): Number of trajectories to generate.

    Returns:
        dict: Trajectories for center and size phases {'center': [], 'size': []}.
    """
    trajectories = {'center': [], 'size': []}
    for _ in range(num_trajectories):
        obs, _ = env.reset()
        trajectory_center = []
        trajectory_size = []
        target_bbox = env.current_gt_bboxes[env.current_gt_index]
        
        while True:
            if env.phase == 'center':
                action = center_agent.expert_agent_action_selection(use_ground_truth=True, target_bbox=target_bbox)
                new_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                trajectory_center.append(Transition(obs, action, reward, done, new_obs))
                obs = new_obs
                if env.phase == 'size' or done:
                    trajectories['center'].extend(trajectory_center)
                    trajectory_center = []
            else:
                action = size_agent.expert_agent_action_selection(use_ground_truth=True, target_bbox=target_bbox)
                new_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                trajectory_size.append(Transition(obs, action, reward, done, new_obs))
                obs = new_obs
                if done:
                    trajectories['size'].extend(trajectory_size)
                    break
            if done:
                break
    return trajectories

def train_il_model(env, trajectories, phase='center', epochs=100):
    """
    Train IL model using expert trajectories.

    Args:
        env (DetectionEnv): Environment instance.
        trajectories (list): Expert transitions.
        phase (str): Phase ('center' or 'size').
        epochs (int): Number of training epochs.

    Returns:
        ILModel: Trained IL model.
    """
    ninputs = env.get_state().shape[1]
    noutputs = env.action_space.n  # NUMBER_OF_ACTIONS_CENTER or NUMBER_OF_ACTIONS_SIZE
    model = ILModel(ninputs, noutputs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for t in trajectories:
            state = torch.tensor(t.state, dtype=torch.float32).unsqueeze(0).to(device)
            action = torch.tensor(t.action, dtype=torch.long).to(device)
            pred = model(state)
            loss = criterion(pred, action)
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
        target_bbox = env.current_gt_bboxes[env.current_gt_index]   # Không dùng nhưng có thể giữ lại nếu muốn thêm vấn đề IOU
        while True:
            model = il_model_center if env.phase == 'center' else il_model_size
            with torch.no_grad():
                state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action_probs = model(state)
                action = torch.argmax(action_probs, dim=1).item()
            new_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.append(Transition(obs, action, reward, done, new_obs))
            obs = new_obs
            if done:
                break
