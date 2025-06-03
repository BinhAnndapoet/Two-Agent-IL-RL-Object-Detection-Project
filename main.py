import os
import torch
import numpy as np
from utils import ReplayBuffer, env_config, calculate_map
from dataset import load_pascal_voc
from env import DetectionEnv
from imitation_learning import initialize_replay_buffer
from agents import CenterDQNAgent, SizeDQNAgent

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Define paths and configurations
    root_dir = "./VOC2012"  # Adjust to your VOC2012 dataset path
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load and split dataset
    print("Loading PASCAL VOC 2012 dataset...")
    datasets, class_names = load_pascal_voc(root_dir, train_ratio=0.8, il_ratio=0.2)
    env_config['dataset'] = datasets
    env_config['current_class'] = class_names
    env_config['n_classes'] = len(class_names)

    # Initialize environment
    env = DetectionEnv(env_config)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer()

    # Initialize IL and generate expert trajectories
    print("Generating expert trajectories and training IL models...")
    replay_buffer, center_il_model, size_il_model = initialize_replay_buffer(
        env, None, None, num_trajectories=env_config['num_il_trajectories']
    )

    # Initialize DQN agents
    print("Initializing DQN agents...")
    center_agent = CenterDQNAgent(
        env=env,
        replay_buffer=replay_buffer,
        target_update_freq=env_config['target_update_freq'],
        exploration_mode=env_config['exploration_mode'],
        n_classes=env_config['n_classes']
    )
    size_agent = SizeDQNAgent(
        env=env,
        replay_buffer=replay_buffer,
        target_update_freq=env_config['target_update_freq'],
        exploration_mode=env_config['exploration_mode'],
        n_classes=env_config['n_classes']
    )

    # Load IL weights into DQN agents (pre-training)
    if center_il_model is not None:
        center_agent.policy_net.load_state_dict(center_il_model.state_dict())
        center_agent.target_net.load_state_dict(center_il_model.state_dict())
    if size_il_model is not None:
        size_agent.policy_net.load_state_dict(size_il_model.state_dict())
        size_agent.target_net.load_state_dict(size_il_model.state_dict())

    # Train DQN agents
    print("Training Center DQN Agent...")
    center_agent.train(max_episodes=env_config['epochs'])
    print("Training Size DQN Agent...")
    size_agent.train(max_episodes=env_config['epochs'])

    # Evaluate on validation set
    print("Evaluating on validation set...")
    env.current_class = 'val'
    center_agent.test(file_path=output_dir, video_filename="center_test.mp4", num_episodes=10)
    size_agent.test(file_path=output_dir, video_filename="size_test.mp4", num_episodes=10)

    # Calculate and print final metrics
    val_dataset = datasets['val']
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    gt_boxes = []
    gt_labels = []

    for img_id, (img, bboxes, labels) in val_dataset.items():
        env.image = img
        env.current_gt_bboxes = bboxes
        env.current_gt_labels = labels
        env.reset()

        # Run inference
        obs, _ = env.reset()
        while True:
            action = center_agent.select_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if info['phase'] == 'size':
                action = size_agent.select_action(obs)
                obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        pred_boxes.extend(info['classification_dictionary']['bbox'])
        pred_labels.extend(info['classification_dictionary']['label'])
        pred_scores.extend(info['classification_dictionary']['confidence'])
        gt_boxes.extend(bboxes)
        gt_labels.extend(labels)

    mAP = calculate_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    print(f"Final mAP@0.5: {mAP:.4f}")

    # Save final models
    center_agent.save(path=os.path.join(output_dir, "center_dqn"))
    size_agent.save(path=os.path.join(output_dir, "size_dqn"))
    print(f"Models saved to {output_dir}")

if __name__ == "__main__":
    main()
