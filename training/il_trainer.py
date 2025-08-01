# --- START OF FILE training/il_trainer.py ---
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random

from tqdm import tqdm
from config import device, env_config, NUM_IL_EPOCHS, BATCH_SIZE
from utils.replay_buffer import ReplayBuffer, Transition
from utils.metrics import calculate_iou
from models.networks import ILModel
from .policies import select_expert_action_center, select_expert_action_size

def generate_expert_trajectory(env, num_trajectories):
    center_trajectories, size_trajectories = [], []
    success_count = 0
    
    print(f"Running num trajectories: {num_trajectories * 5}...")
    for i in tqdm(range(num_trajectories * 5), desc="Generating Expert Trajectories"): # Try more times to get enough successful ones
        if success_count >= num_trajectories:
            break
        
        obs, _ = env.reset()
        for gt_idx in range(len(env.current_gt_bboxes)):
            env.detected_objects.clear()
            target_bbox = env.current_gt_bboxes[gt_idx]
            target_label = env.current_gt_labels[gt_idx]
            env.target_bbox = target_bbox
            env.phase = "center"
            env._update_spaces()
            obs = env.get_state()
            
            single_center_traj = []
            info = {} # Khởi tạo info để tránh lỗi tham chiếu
            
            # Center Phase
            for _ in range(env.max_steps // 2):
                pos_action, class_action, conf, done = select_expert_action_center(env, env.bbox, target_bbox, target_label)
                action = (pos_action, class_action)
                new_obs, _, terminated, truncated, info = env.step(action)
                # Dùng flatten() để đảm bảo obs luôn là vector 1D
                single_center_traj.append((obs.flatten(), pos_action, class_action, conf, done))
                obs = new_obs
                if info.get('phase') == 'size': # Dùng .get() để an toàn hơn
                    break
                if terminated or truncated:
                    break
            
            if single_center_traj:
                center_trajectories.extend(single_center_traj)
            
            # Size Phase (only if center was successful)
            if info.get('phase') == 'size':
                center_trajectories.extend(single_center_traj)
                single_size_traj = []
                for _ in range(env.max_steps // 2):
                    size_action = select_expert_action_size(env, env.bbox, target_bbox)
                    new_obs, _, terminated, truncated, _ = env.step(size_action)
                    conf = 1.0 if calculate_iou(env.bbox, target_bbox) > 0.7 else 0.5
                    single_size_traj.append((obs.squeeze(0), size_action, conf))
                    obs = new_obs
                    if terminated or truncated:
                        break
                size_trajectories.extend(single_size_traj)
                success_count += 1
                print(f"Generated expert trajectory #{success_count}")

    print(f"Generated {len(center_trajectories)} center steps and {len(size_trajectories)} size steps.")
    return center_trajectories, size_trajectories

def train_il_model(env, trajectories, phase='center', epochs=NUM_IL_EPOCHS, batch_size=BATCH_SIZE):
    if not trajectories:
        print(f"No trajectories for phase {phase}, skipping IL training.")
        return None

    old_phase = env.phase
    env.phase = phase
    env._update_spaces()
    ninputs = env.get_state().flatten().shape[0]
    
    model = ILModel(ninputs, phase, n_classes=env.n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=env_config["alpha"])
    pos_criterion = nn.SmoothL1Loss()
    class_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        random.shuffle(trajectories)
        total_loss = 0
        for i in range(0, len(trajectories), batch_size):
            batch = trajectories[i:i+batch_size]
            states = torch.stack([torch.tensor(t[0], dtype=torch.float32) for t in batch]).to(device)
            optimizer.zero_grad()
            
            if phase == 'center':
                pos_actions = torch.tensor([t[1] for t in batch], dtype=torch.long).to(device)
                class_actions = torch.tensor([t[2] for t in batch], dtype=torch.long).to(device)
                confs = torch.tensor([t[3] for t in batch], dtype=torch.float32).to(device)
                dones = torch.tensor([t[4] for t in batch], dtype=torch.float32).to(device)
                pos_pred, class_pred, conf_pred, done_pred = model(states)
                
                L_pos = pos_criterion(pos_pred.gather(1, pos_actions.unsqueeze(1)), torch.ones_like(pos_pred.gather(1, pos_actions.unsqueeze(1))))
                L_class = class_criterion(class_pred, class_actions)
                L_conf = bce_criterion(conf_pred.squeeze(-1), confs)
                L_done = bce_criterion(done_pred.squeeze(-1), dones)
                loss = 0.4 * L_pos + 0.3 * L_class + 0.15 * L_conf + 0.15 * L_done
            else: # size
                size_actions = torch.tensor([t[1] for t in batch], dtype=torch.long).to(device)
                confs = torch.tensor([t[2] for t in batch], dtype=torch.float32).to(device)
                size_pred, conf_pred = model(states)

                L_size = pos_criterion(size_pred.gather(1, size_actions.unsqueeze(1)), torch.ones_like(size_pred.gather(1, size_actions.unsqueeze(1))))
                L_conf = bce_criterion(conf_pred.squeeze(-1), confs)
                loss = 0.6 * L_size + 0.4 * L_conf
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / max(1, len(trajectories) // batch_size)
        if (epoch + 1) % 10 == 0:
            print(f"IL Epoch {epoch+1}/{epochs}, Phase: {phase}, Avg Loss: {avg_loss:.4f}")
    
    env.phase = old_phase # Restore env phase
    env._update_spaces()
    return model

def initialize_replay_buffer(env, num_trajectories):
    center_buffer = ReplayBuffer()
    size_buffer = ReplayBuffer()
    
    print("Generating expert trajectories for IL...")
    center_traj, size_traj = generate_expert_trajectory(env, num_trajectories)
    
    # Fill buffers
    for t in center_traj:
        obs, pos_action, class_action, conf, done = t
        action = pos_action if pos_action < 4 else (4 if done else 5)
        reward = conf * done # Simplified reward for buffer
        center_buffer.append(Transition(obs, action, reward, done, obs))

    for t in size_traj:
        obs, size_action, conf = t
        reward = conf
        done = 1.0 if size_action == 4 else 0.0
        size_buffer.append(Transition(obs, size_action, reward, done, obs))
        
    print("Training IL models...")
    center_model = train_il_model(env, center_traj, phase='center')
    size_model = train_il_model(env, size_traj, phase='size')
    
    return center_buffer, size_buffer, center_model, size_model
# --- END OF FILE training/il_trainer.py ---