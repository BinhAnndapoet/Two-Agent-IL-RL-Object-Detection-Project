import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from env import DetectionEnv
from utils import Transition, ReplayBuffer, calculate_iou, device

class ILModel(nn.Module):
    def __init__(self, input_dim, phase="center", n_classes=20):
        super(ILModel, self).__init__()
        self.phase = phase
        self.n_classes = n_classes
        hidden_dim1 = 256
        hidden_dim2 = 128
        hidden_dim3 = 64

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU()
        )

        if self.phase == "center":
            self.pos_head = nn.Linear(hidden_dim3, 4)
            self.class_head = nn.Linear(hidden_dim3, n_classes)
            self.conf_head = nn.Linear(hidden_dim3, 1)
            self.done_head = nn.Linear(hidden_dim3, 1)
        else:
            self.size_head = nn.Linear(hidden_dim3, 4)
            self.conf_head = nn.Linear(hidden_dim3, 1)

    def forward(self, x):
        x = self.backbone(x)
        if self.phase == "center":
            pos = self.pos_head(x)
            class_probs = self.class_head(x)
            conf = self.conf_head(x)
            done = self.done_head(x)
            return pos, class_probs, conf, done
        else:
            size = self.size_head(x)
            conf = self.conf_head(x)
            return size, conf

def select_expert_action_center(env, current_bbox, target_bbox, target_label):
    current_center = [(current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2]
    target_center = [(target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2]
    dx = target_center[0] - current_center[0]
    dy = target_center[1] - current_center[1]
    alpha_w = env.alpha * (current_bbox[2] - current_bbox[0])
    alpha_h = env.alpha * (current_bbox[3] - current_bbox[1])

    conf = 1.0
    done = 1.0 if abs(dx) < alpha_w / 2 and abs(dy) < alpha_h / 2 else 0.0
    class_action = env.get_class_names().index(target_label) if target_label in env.get_class_names() else 0

    if done:
        pos_action = 4
    else:
        actions = [
            (0, dx > alpha_w),
            (1, dx < -alpha_w),
            (2, dy < -alpha_h),
            (3, dy > alpha_h)
        ]
        valid_actions = [a[0] for a in actions if a[1]]
        pos_action = random.choice(valid_actions) if valid_actions else 4

    return pos_action, class_action, conf, done

def select_expert_action_size(env, current_bbox, target_bbox):
    current_width = current_bbox[2] - current_bbox[0]
    current_height = current_bbox[3] - current_bbox[1]
    target_width = target_bbox[2] - target_bbox[0]
    target_height = target_bbox[3] - target_bbox[1]
    alpha_w = env.alpha * current_width
    alpha_h = env.alpha * current_height

    conf = 1.0
    if (abs(current_width - target_width) < alpha_w and 
        abs(current_height - target_height) < alpha_h):
        size_action = 4
    else:
        actions = [
            (0, current_width < target_width and current_height < target_height),
            (1, current_width > target_width and current_height > target_height),
            (2, current_height > target_height),
            (3, current_width > target_width)
        ]
        valid_actions = [a[0] for a in actions if a[1]]
        size_action = random.choice(valid_actions) if valid_actions else 4

    return size_action

def generate_expert_trajectory(center_agent, size_agent, env, replay_buffer, num_trajectories=1000):
    trajectories = {'center': [], 'size': []}
    for _ in range(num_trajectories):
        obs, _ = env.reset()
        for gt_idx in range(len(env.current_gt_bboxes)):
            env.detected_objects.clear()
            target_bbox = env.current_gt_bboxes[gt_idx]
            target_label = env.current_gt_labels[gt_idx]
            env.current_gt_index = gt_idx
            env.target_bbox = target_bbox
            env.phase = "center"
            env._update_spaces()
            obs = env.get_state()

            while True:
                if env.phase == 'center':
                    pos_action, class_action, conf, done = select_expert_action_center(env, env.bbox, target_bbox, target_label)
                    action = (pos_action, class_action)
                    new_obs, reward, terminated, truncated, info = env.step(action)
                    done_flag = terminated or truncated
                    trajectories['center'].append((obs, pos_action, class_action, conf, done))
                    replay_buffer.append(Transition(obs, pos_action, reward, done_flag, new_obs))
                    obs = new_obs
                    if info["phase"] == 'size' or done_flag:
                        if info["phase"] == 'size':
                            env.step((4, None))
                else:
                    size_action = select_expert_action_size(env, env.bbox, target_bbox)
                    action = size_action
                    conf = 1.0 if size_action == 4 else 0.0
                    new_obs, reward, terminated, truncated, info = env.step(action)
                    done_flag = terminated or truncated
                    trajectories['size'].append((obs, size_action, conf))
                    replay_buffer.append(Transition(obs, action, reward, done_flag, new_obs))
                    obs = new_obs
                    if done_flag:
                        break
                if done_flag:
                    break
    return trajectories

def train_il_model(env, trajectories, phase='center', epochs=100, batch_size=64):
    if not trajectories[phase]:
        print(f"No trajectories for phase {phase}, skipping training.")
        return None
    ninputs = env.get_state().shape[1]
    n_classes = len(env.get_class_names())
    model = ILModel(ninputs, phase, n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    pos_criterion = nn.SmoothL1Loss()
    class_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(trajectories[phase])
        for i in range(0, len(trajectories[phase]), batch_size):
            batch = trajectories[phase][i:i+batch_size]
            states = torch.tensor([t[0] for t in batch], dtype=torch.float32).to(device)
            assert states.shape[1] == ninputs, f"State shape {states.shape} does not match expected input_dim {ninputs}"
            optimizer.zero_grad()
            
            if phase == 'center':
                pos_actions = torch.tensor([t[1] for t in batch], dtype=torch.long).to(device)
                class_actions = torch.tensor([t[2] for t in batch], dtype=torch.long).to(device)
                confs = torch.tensor([t[3] for t in batch], dtype=torch.float32).to(device)
                dones = torch.tensor([t[4] for t in batch], dtype=torch.float32).to(device)
                pos_pred, class_pred, conf_pred, done_pred = model(states)
                
                # Position loss
                valid_pos = (pos_actions >= 0) & (pos_actions < 4)
                L_pos = pos_criterion(
                    pos_pred[valid_pos].gather(1, pos_actions[valid_pos].unsqueeze(1)),
                    torch.ones_like(pos_pred[valid_pos].gather(1, pos_actions[valid_pos].unsqueeze(1)))
                ) if valid_pos.any() else torch.tensor(0.0, device=device)
                
                # Classification loss
                valid_class = (class_actions >= 0) & (class_actions < n_classes)
                L_class = class_criterion(class_pred[valid_class], class_actions[valid_class]) if valid_class.any() else torch.tensor(0.0, device=device)
                
                # Confidence loss
                L_conf = bce_criterion(conf_pred.squeeze(), confs)
                
                # Done loss
                L_done = bce_criterion(done_pred.squeeze(), dones)
                
                # Total loss
                loss = 0.4 * L_pos + 0.3 * L_class + 0.15 * L_conf + 0.15 * L_done
            else:
                size_actions = torch.tensor([t[1] for t in batch], dtype=torch.long).to(device)
                confs = torch.tensor([t[2] for t in batch], dtype=torch.float32).to(device)
                size_pred, conf_pred = model(states)
                
                # Size loss
                valid_size = (size_actions >= 0) & (size_actions < 4)
                L_size = pos_criterion(
                    size_pred[valid_size].gather(1, size_actions[valid_size].unsqueeze(1)),
                    torch.ones_like(size_pred[valid_size].gather(1, size_actions[valid_size].unsqueeze(1)))
                ) if valid_size.any() else torch.tensor(0.0, device=device)
                
                # Confidence loss
                L_conf = bce_criterion(conf_pred.squeeze(), confs)
                
                # Total loss
                loss = 0.6 * L_size + 0.4 * L_conf

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / (len(trajectories[phase]) / batch_size)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Phase: {phase}, Average Loss: {avg_loss:.4f}")

    return model

def initialize_replay_buffer(env, center_agent, size_agent, num_trajectories=1000):
    replay_buffer = ReplayBuffer()
    trajectories = generate_expert_trajectory(center_agent, size_agent, env, replay_buffer, num_trajectories)
    center_model = train_il_model(env, trajectories, phase='center')
    size_model = train_il_model(env, trajectories, phase='size')
    return replay_buffer, center_model, size_model
