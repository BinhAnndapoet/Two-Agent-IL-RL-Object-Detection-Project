# main.py (Updated for robust training loop)

import os
import torch
import numpy as np

# Import các thành phần
from config import env_config
from data.dataset import load_pascal_voc
from environment.env import DetectionEnv
from training.il_trainer import initialize_replay_buffer
from agents.center_agent import CenterDQNAgent
from agents.size_agent import SizeDQNAgent
from utils.metrics import calculate_map
from utils.replay_buffer import Transition

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    root_dir = "./VOC2012"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # --- Giai đoạn 1: Chuẩn bị và Học Bắt chước (IL) ---
    print("================ STAGE 1: PREPARATION & IMITATION LEARNING ================")
    datasets, class_names = load_pascal_voc(root_dir, il_ratio=0.2)
    env_config['dataset'] = datasets
    env_config['current_class'] = class_names
    
    env_il = DetectionEnv(env_config)
    center_buffer, size_buffer, center_il_model, size_il_model = initialize_replay_buffer(
        env_il, num_trajectories=env_config['num_il_trajectories']
    )

    # --- Giai đoạn 2: Chuẩn bị cho Huấn luyện DQN ---
    print("\n================ STAGE 2: DQN AGENT SETUP ================")
    env_config['phase'] = 'dqn'
    env = DetectionEnv(env_config)
    
    center_agent = CenterDQNAgent(env=env, replay_buffer=center_buffer)
    size_agent = SizeDQNAgent(env=env, replay_buffer=size_buffer)

    if center_il_model:
        center_agent.policy_net.load_state_dict(center_il_model.state_dict())
        center_agent.target_net.load_state_dict(center_il_model.state_dict())
        print("[SUCCESS] Loaded IL weights into Center DQN Agent.")
    if size_il_model:
        size_agent.policy_net.load_state_dict(size_il_model.state_dict())
        size_agent.target_net.load_state_dict(size_il_model.state_dict())
        print("[SUCCESS] Loaded IL weights into Size DQN Agent.")

    # --- Giai đoạn 3: Vòng lặp Huấn luyện DQN ---
    print("\n================ STAGE 3: DQN TRAINING LOOP ================")
    max_episodes = env_config['epochs']
    
    for episode in range(max_episodes):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        step_count = 0

        while not (done or truncated):
            current_phase = env.phase
            
            if current_phase == "center":
                action = center_agent.select_action(obs)
                # Đảm bảo action là tuple để env.step xử lý
                if not isinstance(action, tuple): action = (action, None)
                
                new_obs, reward, done, truncated, info = env.step(action)
                
                # Lưu vào buffer. Action lưu là pos_action để tương thích với DQN update
                pos_action, _ = action
                center_agent.replay_buffer.append(Transition(obs, pos_action, reward, done, new_obs))
                center_agent.update()
            else: # 'size'
                action = size_agent.select_action(obs)
                new_obs, reward, done, truncated, info = env.step(action)
                
                size_agent.replay_buffer.append(Transition(obs, action, reward, done, new_obs))
                size_agent.update()
            
            obs = new_obs
            episode_reward += reward
            step_count += 1
        
        # Cập nhật epsilon cho cả hai agent sau mỗi episode
        center_agent.update_epsilon()
        size_agent.update_epsilon()

        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{max_episodes} | Steps: {step_count} | Total Reward: {episode_reward:.3f} | Epsilon: {center_agent.epsilon:.3f}")

    # --- Giai đoạn 4: Đánh giá cuối cùng ---
    print("\n================ STAGE 4: FINAL EVALUATION ================")
    env_config['phase'] = 'test'
    env = DetectionEnv(env_config)
    
    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []
    all_gt_boxes, all_gt_labels = [], []
    
    test_loader = datasets['test'][1]
    num_test_images = len(test_loader.dataset)
    print(f"Evaluating on {num_test_images} test images...")

    for i, (_, _, gt_bboxes, gt_labels) in enumerate(test_loader):
        obs, info = env.reset() # env.reset() sẽ tự động tải ảnh tiếp theo
        done = truncated = False
        while not (done or truncated):
            if env.phase == "center":
                action = center_agent.select_action(obs)
                if not isinstance(action, tuple): action = (action, None)
            else:
                action = size_agent.select_action(obs)
            obs, _, done, truncated, info = env.step(action)
        
        # Thu thập kết quả
        all_pred_boxes.extend(info.get('classification_dictionary', {}).get('bbox', []))
        all_pred_labels.extend(info.get('classification_dictionary', {}).get('label', []))
        all_pred_scores.extend(info.get('classification_dictionary', {}).get('confidence', []))
        all_gt_boxes.extend(gt_bboxes[0])
        all_gt_labels.extend(gt_labels[0])

        if (i + 1) % 100 == 0:
            print(f"  ... processed {i+1}/{num_test_images} images")

    # Tính mAP
    mAP = calculate_map(all_pred_boxes, all_pred_labels, all_pred_scores, all_gt_boxes, all_gt_labels)
    print(f"\nFINAL RESULT: mAP@0.5 = {mAP:.4f}")

    # Lưu model cuối cùng
    center_agent.save(path=os.path.join(output_dir, "center_dqn_final"))
    size_agent.save(path=os.path.join(output_dir, "size_dqn_final"))
    print(f"\nFinal models saved to '{output_dir}'")
    print("\n========================= RUN COMPLETED =========================")

if __name__ == "__main__":
    main()