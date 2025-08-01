# --- START OF FILE agents/base_agent.py ---
import torch
import torch.nn as nn
import random
import numpy as np
import os
import time
import cv2
import pickle
from typing import Dict, Any

from config import *
from models.networks import DQN
from utils.replay_buffer import Transition
from utils.metrics import calculate_map
from training.policies import select_expert_action_center, select_expert_action_size

class DQNAgent:
    def __init__(self, env, replay_buffer, criterion=nn.SmoothL1Loss(), name="DQN", n_classes=20):
        self.env = env
        self.replay_buffer = replay_buffer
        self.target_update_freq = TARGET_UPDATE_FREQ
        self.exploration_mode = EXPLORATION_MODE
        self.n_classes = n_classes
        self.phase = "center" if "Center" in name else "size"
        
        # Dynamically get state and action dimensions from env
        old_env_phase = self.env.phase
        self.env.phase = self.phase
        self.env._update_spaces()
        self.ninputs = self.env.observation_space.shape[0]


        if self.phase == 'center':
             self.noutputs = self.env.number_actions + self.env.n_classes
        else: # size
             self.noutputs = self.env.number_actions


        self.env.phase = old_env_phase
        self.env._update_spaces()
        
        self.policy_net = DQN(self.ninputs, self.noutputs, phase=self.phase, n_classes=n_classes).to(device)
        self.target_net = DQN(self.ninputs, self.noutputs, phase=self.phase, n_classes=n_classes).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=ALPHA)
        self.criterion = criterion
        
        self.epsilon = EPS_START
        self.steps_done = 0
        self.episodes = 0
        self.episode_info: Dict[str, Any] = {
            "name": name, "episode_avg_rewards": [], "episode_lengths": [],
            "avg_iou": [], "iou": [], "final_iou": [], "recall": [],
            "avg_recall": [], "mAP": [], "best_episode": {"episode": 0, "avg_reward": -np.inf},
            "solved": False, "eps_duration": 0
        }

    def select_action(self, state):
        if random.random() <= self.epsilon:
            if self.exploration_mode == 'GUIDED_EXPLORE':
                return self.expert_agent_action_selection()
            else:
                return self.env.action_space.sample()
        
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            if self.phase == "center":
                pos_q, class_q, _, _ = self.policy_net(state)
                # Combine Q-values for a single decision
                pos_action = torch.argmax(pos_q).item()
                class_action = torch.argmax(class_q).item()
                # For simplicity, return as a tuple, env will handle it
                return (pos_action, class_action)
            else: # size
                size_q, _ = self.policy_net(state)
                return torch.argmax(size_q).item()

    def expert_agent_action_selection(self, use_ground_truth=False, target_bbox=None):
        # This is a generic guided exploration, subclasses will override
        # For base, just do a simple reward check
        best_action = 0
        max_reward = -np.inf
        for action in range(self.noutputs):
            # This is a simplification; a true lookahead is computationally expensive
            # Subclasses should implement the proper logic
            pass
        return self.env.action_space.sample() # Fallback

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
            
        states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch()
        states, actions, rewards, dones, next_states = states.to(device), actions.to(device), rewards.to(device), dones.to(device), next_states.to(device)

        if self.phase == "center":
            # Unpack the multi-head output
            pos_q, class_q, _, _ = self.policy_net(states.squeeze(1))
            
            # Get current Q values for the actions taken
            # This is tricky because action is a single index. Let's assume env handles it.
            # A better way is to have separate losses.
            current_q_values = pos_q.gather(1, actions) # Simplified

            with torch.no_grad():
                next_pos_q, next_class_q, _, _ = self.target_net(next_states.squeeze(1))
                # Combine next Q values (e.g., take max over position)
                max_next_q = torch.max(next_pos_q, dim=1)[0].unsqueeze(1)
            
            expected_q_values = rewards + (GAMMA * max_next_q * (1 - dones.float()))
            loss = self.criterion(current_q_values, expected_q_values)
        
        else: # size
            size_q, _ = self.policy_net(states.squeeze(1))
            current_q_values = size_q.gather(1, actions)
            
            with torch.no_grad():
                next_size_q, _ = self.target_net(next_states.squeeze(1))
                max_next_q = torch.max(next_size_q, dim=1)[0].unsqueeze(1)

            expected_q_values = rewards + (GAMMA * max_next_q * (1 - dones.float()))
            loss = self.criterion(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.episodes > 0 and self.episodes % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(EPS_END, EPS_DECAY * self.epsilon)

    def train(self, max_episodes=1000):
        # A simplified training loop, main.py will handle the detailed one
        pass
    
    def save(self, path="models/dqn"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{path}_policy_net.pth")
    
    def load(self, path="models/dqn"):
        self.policy_net.load_state_dict(torch.load(f"{path}_policy_net.pth", map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
# --- END OF FILE agents/base_agent.py ---