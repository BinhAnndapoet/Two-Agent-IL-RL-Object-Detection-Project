from utils import *
from models import *
from imitation_learning import select_expert_action_center, select_expert_action_size

import torch
import torch.nn as nn
import random
import numpy as np
import os
import time
import cv2

class DQN(nn.Module):
    def __init__(self, input_dim, n_outputs, phase="center", n_classes=20):
        super(DQN, self).__init__()
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
            pos_q = self.pos_head(x)
            class_q = self.class_head(x)
            conf_q = self.conf_head(x)
            done_q = self.done_head(x)
            return pos_q, class_q, conf_q, done_q
        else:
            size_q = self.size_head(x)
            conf_q = self.conf_head(x)
            return size_q, conf_q

class DQNAgent:
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ,
                 criterion=nn.SmoothL1Loss(), name="DQN", exploration_mode=EXPLORATION_MODE, n_classes=20):
        self.env = env
        self.replay_buffer = replay_buffer
        self.target_update_freq = target_update_freq
        self.exploration_mode = exploration_mode
        self.n_classes = n_classes
        self.phase = "center" if name == "CenterDQN" else "size"
        self.ninputs = env.get_state().shape[1]
        self.n_pos_outputs = env.action_space.n if self.phase == "size" else 4  # Chỉnh sửa
        self.policy_net = DQN(self.ninputs, self.noutputs, phase=self.phase, n_classes=n_classes).to(device)
        self.target_net = DQN(self.ninputs, self.noutputs, phase=self.phase, n_classes=n_classes).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=ALPHA)
        self.criterion_pos = criterion
        self.criterion_class = nn.CrossEntropyLoss()
        self.epsilon = EPS_START
        self.steps_done = 0
        self.episodes = 0
        self.episode_info = {
            "name": name,
            "episode_avg_rewards": [],
            "episode_lengths": [],
            "avg_iou": [],
            "iou": [],
            "final_iou": [],
            "recall": [],
            "avg_recall": [],
            "mAP": [],
            "best_episode": {"episode": 0, "avg_reward": np.NINF},
            "solved": False,
            "eps_duration": 0
        }
        self.display_every_n_episodes = 100
        self.save_every_n_episodes = 1000

    def select_action(self, state):
        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array")
        if state.shape != (1, self.ninputs):
            raise ValueError(f"State shape {state.shape} does not match expected (1, {self.ninputs})")

        if random.random() <= self.epsilon:
            if self.exploration_mode == GUIDED_EXPLORE:
                action = self.expert_agent_action_selection(use_ground_truth=False)
            else:
                action = self.env.action_space.sample()
            if self.phase == "center":
                pos_action, class_action = action
                return pos_action, class_action if pos_action < 4 else None
            return action, None
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                if self.phase == "center":
                    pos_q, class_q, conf_q, done_q = self.policy_net(state)
                    pos_action = torch.argmax(pos_q, dim=1).item()
                    class_action = torch.argmax(class_q, dim=1).item()
                    conf_action = 5 if conf_q.item() > 0 else None
                    done_action = 4 if done_q.item() > 0 else None
                    if done_action is not None:
                        return done_action, None
                    elif conf_action is not None:
                        return conf_action, None
                    else:
                        return pos_action, class_action
                else:
                    size_q, conf_q = self.policy_net(state)
                    size_action = torch.argmax(size_q, dim=1).item()
                    conf_action = 5 if conf_q.item() > 0 else None
                    return conf_action if conf_action is not None else size_action, None

    def expert_agent_action_selection(self, use_ground_truth=False, target_bbox=None):
        if use_ground_truth and target_bbox is not None:
            if self.phase == "center":
                target_label = self.env.current_gt_labels[self.env.current_gt_index]
                pos_action, _, _, _ = select_expert_action_center(self.env, self.env.bbox, target_bbox, target_label)
                class_action = self.env.get_class_names().index(target_label)
                return pos_action, class_action
            else:
                return select_expert_action_size(self.env, self.env.bbox, target_bbox), None

        positive_actions = []
        negative_actions = []
        positive_class_actions = []
        negative_class_actions = []
        old_state = self.env.bbox
        target_bboxes = [bbox for idx, bbox in enumerate(self.env.current_gt_bboxes) if idx not in self.env.detected_objects]
        target_labels = [label for idx, label in enumerate(self.env.current_gt_labels) if idx not in self.env.detected_objects]
        for action in range(self.noutputs):
            new_state = self.env.transform_action(action, self.env.phase)
            reward = self.env.calculate_reward([new_state], [old_state], target_bboxes, target_labels, self.env.phase) if action < self.noutputs - 2 else self.env.calculate_trigger_reward([new_state], target_bboxes)
            if self.phase == "center" and action >= 6:
                if reward > 0:
                    positive_class_actions.append(action)
                else:
                    negative_class_actions.append(action)
            else:
                if reward > 0:
                    positive_actions.append(action)
                else:
                    negative_actions.append(action)
        pos_action = random.choice(positive_actions) if positive_actions else random.choice(negative_actions)
        class_action = random.choice(positive_class_actions) if positive_class_actions else random.choice(negative_class_actions) if negative_class_actions else None
        return pos_action, class_action if self.phase == "center" else None

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch()
        expected_shapes = {
            "states": (BATCH_SIZE, 1, self.ninputs),
            "actions": (BATCH_SIZE, 1),
            "rewards": (BATCH_SIZE, 1),
            "dones": (BATCH_SIZE, 1),
            "next_states": (BATCH_SIZE, 1, self.ninputs)
        }
        for name, tensor in [("states", states), ("actions", actions), ("rewards", rewards),
                            ("dones", dones), ("next_states", next_states)]:
            if tensor.shape != expected_shapes[name]:
                raise ValueError(f"{name} has shape {tensor.shape}, expected {expected_shapes[name]}")

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        next_states = next_states.to(device)

        if self.phase == "center":
            pos_q, class_q, conf_q, done_q = self.policy_net(states.squeeze(1))
            with torch.no_grad():
                pos_q_next, class_q_next, conf_q_next, done_q_next = self.target_net(next_states.squeeze(1))
                pos_target = rewards + GAMMA * (1 - dones.type(torch.float32)) * torch.max(pos_q_next, dim=1)[0].unsqueeze(1)
                class_target = rewards + GAMMA * (1 - dones.type(torch.float32)) * torch.max(class_q_next, dim=1)[0].unsqueeze(1)
                conf_target = rewards + GAMMA * (1 - dones.type(torch.float32)) * conf_q_next
                done_target = rewards + GAMMA * (1 - dones.type(torch.float32)) * done_q_next

            pos_mask = actions < 4
            class_mask = (actions >= 6) & (actions < 6 + self.n_classes)
            conf_mask = actions == 5
            done_mask = actions == 4

            pos_loss = self.criterion_pos(pos_q[pos_mask].gather(1, actions[pos_mask]), pos_target[pos_mask]) if pos_mask.any() else torch.tensor(0.0, device=device)
            class_loss = self.criterion_class(class_q[class_mask], (actions[class_mask] - 6).squeeze()) if class_mask.any() else torch.tensor(0.0, device=device)
            conf_loss = self.criterion_pos(conf_q[conf_mask], conf_target[conf_mask]) if conf_mask.any() else torch.tensor(0.0, device=device)
            done_loss = self.criterion_pos(done_q[done_mask], done_target[done_mask]) if conf_mask.any() else torch.tensor(0.0, device=device)

            loss = 0.4 * pos_loss + 0.3 * class_loss + 0.15 * conf_loss + 0.15 * done_loss
        else:
            size_q, conf_q = self.policy_net(states.squeeze(1))
            with torch.no_grad():
                size_q_next, conf_q_next = self.target_net(next_states.squeeze(1))
                size_target = rewards + GAMMA * (1 - dones.type(torch.float32)) * torch.max(size_q_next, dim=1)[0].unsqueeze(1)
                conf_target = rewards + GAMMA * (1 - dones.type(torch.float32)) * conf_q_next

            size_mask = actions < 4
            conf_mask = actions == 5

            size_loss = self.criterion_pos(size_q[size_mask].gather(1, actions[size_mask]), size_target[size_mask]) if size_mask.any() else torch.tensor(0.0, device=device)
            conf_loss = self.criterion_pos(conf_q[conf_mask], conf_target[conf_mask]) if conf_mask.any() else torch.tensor(0.0, device=device)

            loss = 0.6 * size_loss + 0.4 * conf_loss

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.episodes % self.target_update_freq == 0 and self.episodes > 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(EPS_END, EPS_DECAY * self.epsilon)

    def train(self, max_episodes=1000):
        self.policy_net.train()
        self.target_net.eval()
        start_time = time.time()

        for episode in range(max_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_boxes = []
            episode_labels = []
            episode_scores = []

            while True:
                action = self.select_action(obs)
                new_obs, reward, terminated, truncated, info = self.env.step(action)
                self.episode_info["iou"].append(info["iou"])
                self.episode_info["recall"].append(info["recall"])

                episode_reward += reward
                done = terminated or truncated

                transition = Transition(obs, action[0], reward, done, new_obs)
                self.replay_buffer.append(transition)

                obs = new_obs
                self.update()
                self.steps_done += 1

                if done:
                    self.episode_info["final_iou"].append(info["iou"])
                    episode_boxes.extend(info["classification_dictionary"]["bbox"])
                    episode_labels.extend(info["classification_dictionary"]["label"])
                    episode_scores.extend(info["classification_dictionary"]["confidence"])
                    self.replay_buffer.rewards.append(episode_reward)
                    self.update_epsilon()
                    self.episodes += 1
                    self.episode_info["episode_avg_rewards"].append(np.mean(self.replay_buffer.rewards))
                    self.episode_info["episode_lengths"].append(self.steps_done)

                    if self.episode_info["episode_avg_rewards"][-1] > self.episode_info["best_episode"]["avg_reward"]:
                        self.episode_info["best_episode"]["episode"] = self.episodes
                        self.episode_info["best_episode"]["avg_reward"] = self.episode_info["episode_avg_rewards"][-1]

                    avg_iou = np.mean(self.episode_info["iou"][-self.env.step_count:])
                    avg_recall = np.mean(self.episode_info["recall"][-self.env.step_count:])
                    self.episode_info["avg_iou"].append(avg_iou)
                    self.episode_info["avg_recall"].append(avg_recall)

                    mAP = calculate_map(episode_boxes, episode_labels, episode_scores,
                                      self.env.current_gt_bboxes, self.env.current_gt_labels)
                    self.episode_info["mAP"].append(mAP)

                    if self.episodes >= SUCCESS_CRITERIA_EPS:
                        self.episode_info["solved"] = True

                    if self.episodes % self.display_every_n_episodes == 0:
                        print(f"\033[35mEpisode: {self.episodes} Epsilon: {self.epsilon:.2f} "
                              f"Average Reward: {self.episode_info['episode_avg_rewards'][-1]:.2f} "
                              f"Episode Length: {self.episode_info['episode_lengths'][-1]} "
                              f"Average IoU: {avg_iou:.2f} Average Recall: {avg_recall:.2f} "
                              f"mAP@0.5: {mAP:.2f} Final IoU: {self.episode_info['final_iou'][-1]:.2f}\033[0m")

                    if self.episodes % self.save_every_n_episodes == 0:
                        self.save(path=f"models/{self.episode_info['name']}_ep{self.episodes}")

                    self.steps_done = 0
                    break

            if self.episode_info["solved"]:
                print(f"\033[32mCompleted {self.episodes} episodes!\033[0m")
                break
        self.episode_info["eps_duration"] = time.time() - start_time
        self.save(path=f"models/{self.episode_info['name']}_final")

    def test(self, file_path='render', video_filename='output_video.mp4', num_episodes=1):
        self.policy_net.eval()
        self.target_net.eval()
        if os.path.exists(os.path.join(file_path, video_filename)):
            os.remove(os.path.join(file_path, video_filename))
        os.makedirs(file_path, exist_ok=True)
        video_writer = cv2.VideoWriter(
            os.path.join(file_path, video_filename),
            cv2.VideoWriter_fourcc(*'avc1'),
            5,
            (self.env.width, self.env.height)
        )
        test_iou = []
        test_recall = []
        test_boxes = []
        test_labels = []
        test_scores = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            self.env.detected_objects.clear()
            frames = [self.env.render()] if self.env.render_mode else [self.env.display(mode='trigger_image')]
            while True:
                action = self.select_action(obs)
                obs, _, terminated, truncated, info = self.env.step(action)
                frame = self.env.render() if self.env.render_mode else self.env.display(mode='trigger_image')
                frames.append(frame)
                test_iou.append(info["iou"])
                test_recall.append(info["recall"])
                test_boxes.extend(info["classification_dictionary"]["bbox"])
                test_labels.extend(info["classification_dictionary"]["label"])
                test_scores.extend(info["classification_dictionary"]["confidence"])
                if terminated or truncated:
                    break
            frames.append(self.env.display(mode='detection'))
            for frame in frames:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
        mAP = calculate_map(test_boxes, test_labels, test_scores,
                           self.env.current_gt_bboxes, self.env.current_gt_labels)
        print(f'\033[92mVideo saved to: {os.path.join(file_path, video_filename)}\033[0m')
        print(f'\033[92mTest Metrics - Average IoU: {np.mean(test_iou):.2f}, '
              f'Average Recall: {np.mean(test_recall):.2f}, mAP@0.5: {mAP:.2f}\033[0m')
        return {"avg_iou": np.mean(test_iou), "avg_recall": np.mean(test_recall), "mAP": mAP}

    def save(self, path="models/dqn"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{path}_policy_net.pth")
        torch.save(self.target_net.state_dict(), f"{path}_target_net.pth")
        torch.save(self.optimizer.state_dict(), f"{path}_optimizer.pth")
        np.save(f"{path}_episode_info.npy", self.episode_info)

    def load(self, path="models/dqn"):
        for file in ["policy_net.pth", "target_net.pth", "optimizer.pth", "episode_info.npy"]:
            if not os.path.exists(f"{path}_{file}"):
                raise FileNotFoundError(f"Missing file: {path}_{file}")
        self.policy_net.load_state_dict(torch.load(f"{path}_policy_net.pth"))
        self.target_net.load_state_dict(torch.load(f"{path}_target_net.pth"))
        self.policy_net.to(device)
        self.target_net.to(device)
        self.optimizer.load_state_dict(torch.load(f"{path}_optimizer.pth"))
        self.episode_info = np.load(f"{path}_episode_info.npy", allow_pickle=True).item()
        self.epsilon = EPS_END

class CenterDQNAgent(DQNAgent):
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ,
                 criterion=nn.SmoothL1Loss(), exploration_mode=EXPLORATION_MODE, n_classes=20):
        super().__init__(env, replay_buffer, target_update_freq, criterion, "CenterDQN", exploration_mode, n_classes)

    def expert_agent_action_selection(self, use_ground_truth=False, target_bbox=None):
        if use_ground_truth and target_bbox is not None:
            target_label = self.env.current_gt_labels[self.env.current_gt_index]
            pos_action, _, _, _ = select_expert_action_center(self.env, self.env.bbox, target_bbox, target_label)
            class_action = self.env.get_class_names().index(target_label)
            return pos_action, class_action
        return super().expert_agent_action_selection(use_ground_truth=False)

class SizeDQNAgent(DQNAgent):
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ,
                 criterion=nn.SmoothL1Loss(), exploration_mode=EXPLORATION_MODE, n_classes=20):
        super().__init__(env, replay_buffer, target_update_freq, criterion, "SizeDQN", exploration_mode, n_classes)

    def expert_agent_action_selection(self, use_ground_truth=False, target_bbox=None):
        if use_ground_truth and target_bbox is not None:
            return select_expert_action_size(self.env, self.env.bbox, target_bbox), None
        return super().expert_agent_action_selection(use_ground_truth=False)
