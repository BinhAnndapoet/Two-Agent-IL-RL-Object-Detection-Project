import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from utils import Transition, ReplayBuffer, calculate_iou, device, env_config

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
            # [THAY ĐỔI]: Đảm bảo class_head luôn có kích thước đúng
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

# Trong imitation_learning.py - Sửa hàm generate_expert_trajectory
def generate_expert_trajectory(center_agent, size_agent, env, replay_buffer, num_trajectories=None):
    
    if num_trajectories is None:
        num_trajectories = env_config.get("num_il_trajectories", 1000)
    
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

            # Giai đoạn Center
            center_steps = 0
            max_center_steps = 50  # Giới hạn số bước tối đa để tránh vòng lặp vô hạn
            
            while env.phase == "center" and center_steps < max_center_steps:
                pos_action, class_action, conf, done = select_expert_action_center(env, env.bbox, target_bbox, target_label)
                
                # Nếu đã đủ bước và gần mục tiêu, buộc chuyển sang done action
                if center_steps >= env.trigger_steps and calculate_iou(env.bbox, target_bbox) > env.threshold * 0.8:
                    pos_action = 4  # Đây là trigger action để chuyển phase
                    
                action = (pos_action, class_action)
                new_obs, reward, terminated, truncated, info = env.step(action)
                
                trajectories['center'].append((obs, pos_action, class_action, conf, done))
                replay_buffer.append(Transition(obs, pos_action, reward, terminated or truncated, new_obs))
                obs = new_obs
                center_steps += 1
                
                if terminated or truncated:
                    break
                    
            # [SỬA ĐỔI LỖI 1]: Nếu chưa chuyển phase thì ép chuyển sang size
            if env.phase != "size":
                print(f"Force chuyển phase (IoU={calculate_iou(env.bbox, target_bbox):.3f} < threshold) sau {center_steps} steps.")
                env.phase = "size"
                env._update_spaces()
                obs = env.get_state()

            # Giai đoạn Size (nếu chuyển sang)
            if env.phase == "size":
                size_steps = 0
                max_size_steps = 30  # Giới hạn số bước tối đa
                
                # [SỬA ĐỔI LỖI 2]: Đảm bảo lấy state đúng phase
                while env.phase == "size" and size_steps < max_size_steps:
                    size_action = select_expert_action_size(env, env.bbox, target_bbox)
                    action = size_action
                    new_obs, reward, terminated, truncated, info = env.step(action)
                    
                    # [THÊM] Lưu confidence cao hơn nếu IoU tốt
                    current_iou = calculate_iou(env.bbox, target_bbox)
                    conf_value = 1.0 if current_iou > 0.7 else (0.5 if current_iou > 0.5 else 0.1)
                    
                    trajectories['size'].append((obs, size_action, conf_value))
                    replay_buffer.append(Transition(obs, action, reward, terminated or truncated, new_obs))
                    obs = new_obs
                    size_steps += 1
                    
                    if terminated or truncated:
                        break
                
                print(f"Generated size trajectory with {size_steps} steps")
    
    # [THÊM] In thống kê để kiểm tra
    print(f"Generated {len(trajectories['center'])} center trajectories")
    print(f"Generated {len(trajectories['size'])} size trajectories")
    
    return trajectories

def train_il_model(env, trajectories, phase='center', epochs=None, batch_size=None):
    
    if epochs is None:
        epochs = env_config.get("epochs", 100)
    if batch_size is None:
        batch_size = env_config.get("batch_size", 64)
    
    if not trajectories[phase]:
        print(f"No trajectories for phase {phase}, skipping training.")
        return None
    
    # [SỬA ĐỔI LỖI 2]: Luôn cập nhật phase và action space trước khi lấy state
    env.phase = phase
    env._update_spaces()
    
    # phần cũ
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
            states = torch.tensor(np.array([t[0].flatten() for t in batch]), dtype=torch.float32).to(device) # [update]: add .flatten() to t[0] for correct shape
            
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

def initialize_replay_buffer(env, center_agent, size_agent, num_trajectories=None):
    
    if num_trajectories is None:
        num_trajectories = env_config.get("num_il_trajectories", 1000)
    
    replay_buffer = ReplayBuffer()
    trajectories = generate_expert_trajectory(center_agent, size_agent, env, replay_buffer, num_trajectories)
    center_model = train_il_model(env, trajectories, phase='center')
    size_model = train_il_model(env, trajectories, phase='size')
    return replay_buffer, center_model, size_model
