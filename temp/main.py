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
    print("Initializing environment...")
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
    val_dataset, val_loader = datasets['val']
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    gt_boxes = []
    gt_labels = []

    # [update]: đổi ảnh trong DataLoader từ tensor sang numpy array [for -> env.reset()]
    print("Calculating mAP on validation set...")
    for image_ids, image_batch, gt_bbox_batches, gt_label_batches in val_loader:
        image_id, image_tensor, image_gt_boxes, image_gt_labels = (
            image_ids[0], image_batch[0], gt_bbox_batches[0], gt_label_batches[0]
        )  # batch_size=1
        
        # Log image_id để theo dõi (tùy chọn)
        print(f"Processing image: {image_id} with {len(image_gt_boxes)} ground truth boxes")
        
        # Chuyển tensor sang Numpy array & đảo ngược chuẩn hóa để xử ảnh trong env.py: 
            # [render function]: cv2.rectangle(img, ...)
            # [get_state function]: cropped_image = image[y1:y2, x1:x2] -> yêu cầu numpy array để slicing
        
        # Chuyển tensor sang NumPy và đảo ngược chuẩn hóa
        channel_means = np.array([0.485, 0.456, 0.406])[:, None, None]  # [C, 1, 1]
        channel_stds = np.array([0.229, 0.224, 0.225])[:, None, None]  # [C, 1, 1]
        image_array = image_tensor.cpu().numpy()  # [C, H, W]
        image_array = image_array * channel_stds + channel_means  # Đảo ngược chuẩn hóa
        image_array = np.clip(image_array, 0, 1) * 255.0  # Đưa về [0, 255]
        image_array = image_array.transpose(1, 2, 0)  # [H, W, C]
        
        
        env.image = image_array.astype(np.uint8)  # Gán vào env.image
        env.current_gt_bboxes = gt_boxes
        env.current_gt_labels = gt_labels
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
        gt_boxes.extend(image_gt_boxes) # [change]: gt_boxes -> image_gt_boxes
        gt_labels.extend(image_gt_labels) # [change]: gt_labels -> image_gt_labels

    mAP = calculate_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    print(f"Final mAP@0.5: {mAP:.4f}")

    # Save final models
    center_agent.save(path=os.path.join(output_dir, "center_dqn"))
    size_agent.save(path=os.path.join(output_dir, "size_dqn"))
    print(f"Models saved to {output_dir}")

if __name__ == "__main__":
    main()

