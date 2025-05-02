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

class DQNAgent():
    """
    Base DQN agent for interacting with DetectionEnv.

    Args:
        env: DetectionEnv environment.
        replay_buffer: Replay buffer for storing transitions.
        target_update_freq (int): Frequency to update target network.
        criterion: Loss function (default: SmoothL1Loss).
        name (str): Agent name.
        network: Q-network architecture (default: DQN).
        exploration_mode (str): Exploration strategy ('random' or 'guided').

    Attributes:
        env: Environment instance.
        replay_buffer: Replay buffer instance.
        policy_net: Policy Q-network.
        target_net: Target Q-network.
        optimizer: Optimizer for policy network.
        epsilon: Exploration probability.
        episodes: Number of episodes completed.
        episode_info: Training statistics.
    """
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ, 
                 criterion=nn.SmoothL1Loss(), name="DQN", network=DQN, exploration_mode=EXPLORATION_MODE):
        self.env = env
        self.replay_buffer = replay_buffer
        self.target_update_freq = target_update_freq
        self.exploration_mode = exploration_mode
        self.ninputs = env.get_state().shape[1]
        self.noutputs = env.action_space.n
        self.policy_net = network(self.ninputs, self.noutputs).to(device)
        self.target_net = network(self.ninputs, self.noutputs).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=ALPHA)
        self.criterion = criterion
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
            "best_episode": {"episode": 0, "avg_reward": np.NINF},
            "solved": False,
            "eps_duration": 0
        }
        self.display_every_n_episodes = 100
        self.save_every_n_episodes = 1000

    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state.

        Returns:
            int: Selected action.

        Raises:
            ValueError: If state is not a numpy array or has incorrect shape.
        """
        if random.random() <= self.epsilon:
            if self.exploration_mode == GUIDED_EXPLORE:
                action = self.expert_agent_action_selection(use_ground_truth=False)
            else:
                action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                qvalues = self.policy_net(state)
                action = qvalues.argmax().item()
        return action

    def expert_agent_action_selection(self, use_ground_truth=False):
        """
        Select action using expert policy.

        Args:
            use_ground_truth (bool): Whether to use ground truth (True for IL, False for RL).

        Returns:
            int: Expert-selected action.
        """
        positive_actions = []
        negative_actions = []
        old_state = self.env.bbox
        for action in range(self.noutputs):
            new_state = self.env.transform_action(action, self.env.phase)
            reward = self.env.calculate_reward([new_state], [old_state], [], self.env.phase) if action < self.noutputs - 1 else self.env.calculate_trigger_reward([new_state], [])
            if reward > 0:
                positive_actions.append(action)
            else:
                negative_actions.append(action)
        return random.choice(positive_actions) if positive_actions else random.choice(negative_actions)

    def update(self):
        """
        Update policy network using a batch of transitions.

        Raises:
            ValueError: If batch tensors have incorrect shapes.
        """
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

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        next_states = next_states.to(device)
        qvalues = self.policy_net(states.squeeze(1)).gather(1, actions)

        with torch.no_grad():
            target_qvalues = self.target_net(next_states.squeeze(1))
            max_target_qvalues = torch.max(target_qvalues, axis=1).values.unsqueeze(1)
            next_qvalues = rewards + GAMMA * (1 - dones.type(torch.float32)) * max_target_qvalues

        loss = self.criterion(qvalues, next_qvalues)
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.episodes % self.target_update_freq == 0 and self.episodes > 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """
        Update epsilon for exploration.
        """
        self.epsilon = max(EPS_END, EPS_DECAY * self.epsilon)

    def train(self, max_episodes=1000):
        """
        Train the agent.

        Args:
            max_episodes (int): Maximum number of episodes.
        """
        self.policy_net.train()
        self.target_net.eval()
        start_time = time.time()

        for episode in range(max_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0

            while True:
                action = self.select_action(obs)
                new_obs, reward, terminated, truncated, info = self.env.step(action)
                self.episode_info["iou"].append(info["iou"])
                self.episode_info["recall"].append(info["recall"])

                episode_reward += reward
                done = terminated or truncated

                transition = Transition(obs, action, reward, done, new_obs)
                self.replay_buffer.append(transition)

                obs = new_obs
                self.update()
                self.steps_done += 1

                if done:
                    self.episode_info["final_iou"].append(info["iou"])
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

                    if self.episodes >= SUCCESS_CRITERIA_EPS:
                        self.episode_info["solved"] = True

                    if self.episodes % self.display_every_n_episodes == 0:
                        print(f"\033[35mEpisode: {self.episodes} Epsilon: {self.epsilon:.2f} "
                              f"Average Reward: {self.episode_info['episode_avg_rewards'][-1]:.2f} "
                              f"Episode Length: {self.episode_info['episode_lengths'][-1]} "
                              f"Average IoU: {avg_iou:.2f} Average Recall: {avg_recall:.2f} "
                              f"Final IoU: {self.episode_info['final_iou'][-1]:.2f}\033[0m")
                        
                    if self.episodes % self.save_every_n_episodes == 0:
                        self.save(path=f"models/{self.episode_info['name']}_ep{self.episodes}")

                    self.steps_done = 0
                    break
                
            if self.episode_info["solved"]:
                print(f"\033[32mCompleted {self.episodes} episodes!\033[0m")
                break
        self.episode_info["eps_duration"] = time.time() - start_time
        self.save(path=f"models/{self.episode_info['name']}_final")

    def test(self, file_path='render', video_filename='output_video.mp4'):
        """
        Test the trained agent and create an MP4 video.

        Args:
            file_path (str): Path to save video.
            video_filename (str): Video filename.

        Returns:
            dict: Test metrics (average IoU, recall).
        """
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
        obs, _ = self.env.reset()
        frames = [self.env.render()] if self.env.render_mode else [self.env.display(mode='trigger_image')]
        test_iou = []
        test_recall = []
        while True:
            action = int(torch.argmax(self.policy_net(torch.from_numpy(obs).float().unsqueeze(0).to(device))).item())
            obs, _, terminated, truncated, info = self.env.step(action)
            frame = self.env.render() if self.env.render_mode else self.env.display(mode='trigger_image')
            frames.append(frame)
            test_iou.append(info["iou"])
            test_recall.append(info["recall"])
            if terminated or truncated:
                break
        frames.append(self.env.display(mode='detection'))
        for frame in frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
        print(f'\033[92mVideo saved to: {os.path.join(file_path, video_filename)}\033[0m')
        return {"avg_iou": np.mean(test_iou), "avg_recall": np.mean(test_recall)}

    # def save(self, path="models/dqn"):
    #     """
    #     Save the model.

    #     Args:
    #         path (str): Save path.
    #     """
    #     os.makedirs(path, exist_ok=True)
    #     torch.save(self.policy_net.state_dict(), f"{path}/policy_net.pth")
    #     torch.save(self.target_net.state_dict(), f"{path}/target_net.pth")
    #     torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")
    #     np.save(f"{path}/episode_info.npy", self.episode_info)

    # def load(self, path="models/dqn"):
    #     """
    #     Load the model.

    #     Args:
    #         path (str): Load path.

    #     Raises:
    #         FileNotFoundError: If model files are missing.
    #     """
    #     for file in ["policy_net.pth", "target_net.pth", "optimizer.pth", "episode_info.npy"]:
    #         if not os.path.exists(f"{path}/{file}"):
    #             raise FileNotFoundError(f"Missing file: {path}/{file}")
    #     self.policy_net.load_state_dict(torch.load(f"{path}/policy_net.pth"))
    #     self.target_net.load_state_dict(torch.load(f"{path}/target_net.pth"))
    #     self.policy_net.to(device)
    #     self.target_net.to(device)
    #     self.optimizer.load_state_dict(torch.load(f"{path}/optimizer.pth"))
    #     self.episode_info = np.load(f"{path}/episode_info.npy", allow_pickle=True).item()
    #     self.epsilon = EPS_END

class CenterDQNAgent(DQNAgent):
    """
    DQN agent for adjusting bounding box center in the center phase.

    Args:
        env: DetectionEnv environment.
        replay_buffer: Replay buffer.
        target_update_freq (int): Target network update frequency.
        criterion: Loss function.
        exploration_mode (str): Exploration strategy.
    """
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ, criterion=nn.SmoothL1Loss(), exploration_mode=EXPLORATION_MODE):
        super().__init__(env, replay_buffer, target_update_freq, criterion, "CenterDQN", DQN, exploration_mode)

    def expert_agent_action_selection(self, use_ground_truth=False, target_bbox=None):
        """
        Select action using expert policy for center phase.

        Args:
            use_ground_truth (bool): Whether to use ground truth (True for IL, False for RL).
            target_bbox (list, optional): Ground truth bbox [x1, y1, x2, y2]. Defaults to None.

        Returns:
            int: Expert-selected action (0: right, 1: left, 2: up, 3: down, 4: trigger).
        """
        if use_ground_truth:
            if target_bbox is None:
                target_bboxes = self.env.current_gt_bboxes
                ious = [self.env.calculate_iou(self.env.bbox, gt) for gt in target_bboxes]
                target_bbox = target_bboxes[np.argmax(ious)]
            return select_expert_action_center(self.env, self.env.bbox, target_bbox)
        else:
            return super().expert_agent_action_selection(use_ground_truth=False)

class SizeDQNAgent(DQNAgent):
    """
    DQN agent for adjusting bounding box size in the size phase.

    Args:
        env: DetectionEnv environment.
        replay_buffer: Replay buffer.
        target_update_freq (int): Target network update frequency.
        criterion: Loss function.
        exploration_mode (str): Exploration strategy.
    """
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ, criterion=nn.SmoothL1Loss(), exploration_mode=EXPLORATION_MODE):
        super().__init__(env, replay_buffer, target_update_freq, criterion, "SizeDQN", DQN, exploration_mode)

    def expert_agent_action_selection(self, use_ground_truth=False, target_bbox=None):
        """
        Select action using expert policy for size phase.

        Args:
            use_ground_truth (bool): Whether to use ground truth (True for IL, False for RL).
            target_bbox (list, optional): Ground truth bbox [x1, y1, x2, y2]. Defaults to None.

        Returns:
            int: Expert-selected action (0: bigger, 1: smaller, 2: fatter, 3: taller, 4: trigger).
        """
        if use_ground_truth:
            if target_bbox is None:
                target_bboxes = self.env.current_gt_bboxes
                ious = [self.env.calculate_iou(self.env.bbox, gt) for gt in target_bboxes]
                target_bbox = target_bboxes[np.argmax(ious)]
            return select_expert_action_size(self.env, self.env.bbox, target_bbox)
        else:
            return super().expert_agent_action_selection(use_ground_truth=False)
