
from models import *
from utils import *

import torch
import torch.nn as nn
import random
import numpy as np
import os
import time

class DQNAgent():
    """
        The DQN agent that interacts with the environment

        Args:
            env: The environment to interact with
            replay_buffer: The replay buffer to store and sample transitions from
            target_update_freq: The frequency with which the target network is updated (default: TARGET_UPDATE_FREQ)
            criterion: The loss function used to train the policy network (default: nn.SmoothL1Loss())
            name: The name of the agent (default: DQN)
            network: The network used to estimate the action-value function (default: DQN)
            exploration_mode: The exploration mode used by the agent (default: GUIDED_EXPLORE)

        Attributes:
            env: The environment to interact with
            replay_buffer: The replay buffer to store and sample transitions from
            nsteps: The number of steps to run the agent for
            target_update_freq: The frequency with which the target network is updated
            ninputs: The number of inputs
            noutputs: The number of outputs
            policy_net: The policy network
            target_net: The target network
            optimizer: The optimizer used to update the policy network
            criterion: The loss function used to train the policy network
            epsilon: The probability of selecting a random action
            steps_done: The number of steps the agent has run for
            episodes: The number of episodes the agent has run for
            episode_avg_rewards: The average reward for each episode
            episode_lengths: The lengths of each episode
            best_episode: The best episode
            solved: Whether the environment is solved
            display_every_n_episodes: The number of episodes after which the results are displayed
            time: The time taken to run the agent
    """
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ,
                 criterion=nn.SmoothL1Loss(), name="DQN", network=DQN,
                 exploration_mode=EXPLORATION_MODE, noutputs=4):
        self.env = env
        self.replay_buffer = replay_buffer
        self.target_update_freq = target_update_freq
        self.exploration_mode = exploration_mode
        self.ninputs = env.get_state().shape[1]
        self.noutputs = noutputs
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
        self.display_every_n_episodes = 1000000

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state.

        Returns:
            int: Selected action.
        """

        # Selecting a random action with probability epsilon
        if random.random() <= self.epsilon: # Exploration
            if self.exploration_mode == GUIDED_EXPLORE: # Guided exploration
                # Expert agent action selection
                action = self.expert_agent_action_selection()
            else: # Random exploration
                # Normal Random action Selection
                action = self.env.action_space.sample()
        else: # Exploitation
            # Selecting the action with the highest Q-value otherwise
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                qvalues = self.policy_net(state)
                action = qvalues.argmax().item()
        return action

    def expert_agent_action_selection(self):
        """
        Select action using expert policy (to be overridden by subclasses).

        Returns:
            int: Selected action.
        """
        raise NotImplementedError("Subclasses must implement expert_agent_action_selection")

    def update(self):
        """
        Update the policy network using a batch of transitions.
        """

        if len(self.replay_buffer) < BATCH_SIZE:
            return
        # Sampling a batch of transitions from the replay buffer
        states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch()

        # Converting the tensors to cuda tensors
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        next_states = next_states.to(device)

        # Calculating the Q-values for the current states
        qvalues = self.policy_net(states.squeeze(1)).gather(1, actions)

        # Calculating the Q-values for the next states
        with torch.no_grad():
            # Calculating the Q-values for the next states using the target network (Q(s',a'))
            target_qvalues = self.target_net(next_states.squeeze(1))

            # Calculating the maximum Q-values for the next states (max(Q(s',a'))
            max_target_qvalues = torch.max(target_qvalues, axis=1).values.unsqueeze(1)

            # Calculating the next Q-values using the Bellman equation (Q(s,a) = r + γ * max(Q(s',a')))
            next_qvalues = rewards + GAMMA * (1 - dones.type(torch.float32)) * max_target_qvalues

        # Calculating the loss
        loss = self.criterion(qvalues, next_qvalues)

        # Optimizing the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clipping the gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Updating the target network
        if self.episodes % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """
        Update epsilon for exploration.
        """
        self.epsilon = max(EPS_END, EPS_DECAY * self.epsilon)

    def train(self):
        """
        Train the agent over multiple episodes.
        """
        # Setting networks to training mode
        self.policy_net.train()
        self.target_net.train()

        # Resetting the environment
        obs, _ = self.env.reset()

        # Retrieving the starting time
        start_time = time.time()

        # Setting the episode_reward to 0
        episode_reward = 0

        # Running the agent for nsteps steps
        for step in range(MAX_STEPS):
            # Selecting an action
            action = self.select_action(obs)

            # Taking a step in the environment
            new_obs, reward, terminated, truncated, info = self.env.step(action)

            # Adding the IoU and recall to the episode info
            self.episode_info["iou"].append(info["iou"])
            self.episode_info["recall"].append(info["recall"])

            # Adding the reward to the cumulative reward
            episode_reward += reward

            # Setting done to terminated or truncated
            done = terminated or truncated

            # Creating a transition
            transition = Transition(obs, action, reward, done, new_obs)

            # Appending the transition to the replay buffer
            self.replay_buffer.append(transition)

            # Resetting the observation
            obs = new_obs

            # Ending the episode and displaying the results if the episode is done
            if done:
                # Appending the final IoU to the episode info
                self.episode_info["final_iou"].append(info["iou"])

                # Appending the rewards to the replay buffer
                self.replay_buffer.rewards.append(episode_reward)

                # Updating epsilon
                self.update_epsilon()

                # Resetting the environment
                obs, _ = self.env.reset()

                # Incrementing the number of episodes
                self.episodes += 1

                # Appending the average episode reward
                self.episode_info["episode_avg_rewards"].append(np.mean(self.replay_buffer.rewards))

                # Appending the episode length
                self.episode_info["episode_lengths"].append(self.steps_done)

                # Updating the best episode
                if self.episode_info["episode_avg_rewards"][-1] > self.episode_info["best_episode"]["avg_reward"]:
                    self.episode_info["best_episode"]["episode"] = self.episodes
                    self.episode_info["best_episode"]["avg_reward"] = self.episode_info["episode_avg_rewards"][-1]

                # Calculating the average IoU and recall    
                avg_iou = np.mean(self.episode_info["iou"][-self.env.step_count:])
                avg_recall = np.mean(self.episode_info["recall"][-self.env.step_count:])

                # Appending the average IoU and recall
                self.episode_info["avg_iou"].append(avg_iou)
                self.episode_info["avg_recall"].append(avg_recall)

                # If the environment number of episodes is greater than SUCCESS_CRITERIA, the environment is considered solved
                if USE_EPISODE_CRITERIA and self.episodes >= SUCCESS_CRITERIA_EPS:
                    self.episode_info["solved"] = True
                # If the environment number of epochs is greater than SUCCESS_CRITERIA, the environment is considered solved
                elif self.env.epochs >= SUCCESS_CRITERIA_EPOCHS:
                    self.episode_info["solved"] = True

                # Displaying the results
                if self.episodes % self.display_every_n_episodes == 0:
                    print(f"\033[35mEpisode: {self.episodes} Epsilon: {self.epsilon:.2f} "
                          f"Average Reward: {self.episode_info['episode_avg_rewards'][-1]:.2f} "
                          f"Episode Length: {self.episode_info['episode_lengths'][-1]} "
                          f"Average IoU: {avg_iou:.2f} Average Recall: {avg_recall:.2f} "
                          f"Epochs: {self.env.epochs} Final IoU: {self.episode_info['final_iou'][-1]:.2f}\033[0m")
                    
                # Resetting the cumulative reward    
                episode_reward = 0
                self.steps_done = 0

                # Checking if the environment is solved
                if self.episode_info["solved"]:
                    print(f"\033[32mCompleted {self.episodes} episodes!\033[0m")
                    break

            # Updating the policy network
            self.update()

            # Updating the number of steps
            self.steps_done += 1

        # Calculating the time taken
        self.episode_info["eps_duration"] = time.time() - start_time

    def save(self, path="models/dqn"):
        """
        Save the model.

        Args:
            path (str): Path to save the model.
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{path}/policy_net.pth")
        torch.save(self.target_net.state_dict(), f"{path}/target_net.pth")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")
        np.save(f"{path}/episode_info.npy", self.episode_info)

    def load(self, path="models/dqn"):
        """
        Load the model.

        Args:
            path (str): Path to load the model.
        """
        self.policy_net.load_state_dict(torch.load(f"{path}/policy_net.pth"))
        self.target_net.load_state_dict(torch.load(f"{path}/target_net.pth"))
        self.policy_net.to(device)
        self.target_net.to(device)
        self.optimizer.load_state_dict(torch.load(f"{path}/optimizer.pth"))
        self.episode_info = np.load(f"{path}/episode_info.npy", allow_pickle=True).item()
        self.epsilon = EPS_END

class CenterDQNAgent(DQNAgent):
    """
    DQN Agent for adjusting the center of the bounding box.

    Inherits from DQNAgent and overrides expert action selection for center phase.

    Args:
        env: DetectionEnv environment.
        replay_buffer: Replay buffer for storing transitions.
        target_update_freq: Frequency to update target network.
        criterion: Loss function.
        name: Agent name.
        network: Q-network architecture.
        exploration_mode: Exploration strategy.
    """
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ,
                 criterion=nn.SmoothL1Loss(), name="CenterDQN", network=DQN,
                 exploration_mode=EXPLORATION_MODE):
        super().__init__(env, replay_buffer, target_update_freq, criterion, name, network,
                         exploration_mode, noutputs=4)  # 4 actions: right, left, up, down

    def expert_agent_action_selection(self):
        """
        Select action using expert policy based on center distance.

        Returns:
            int: Selected action.
        """

        # Creating lists to hold the positive actions and negative actions
        positive_actions = []
        negative_actions = []

        # Retrieving the bounding box from the environment
        old_state = self.env.bbox

        # Retrieving the target bounding boxes from the environment
        target_bboxes = self.env.current_gt_bboxes

        # Looping through the actions
        for action in range(self.noutputs):  # Actions 0-3
            # Retrieving the new state
            new_state = self.env.transform_action(action)

            # Calculating the reward
            reward = self.env.calculate_reward(new_state, old_state, target_bboxes, phase='center')

            # Appending the action to the positive or negative actions list based on the reward
            if reward > 0:
                positive_actions.append(action)
            else:
                negative_actions.append(action)

        # Returning a random choice from the positive actions if the list is not empty
        return random.choice(positive_actions) if positive_actions else random.choice(negative_actions)

class SizeDQNAgent(DQNAgent):
    """
    DQN Agent for adjusting the size of the bounding box.

    Inherits from DQNAgent and overrides expert action selection for size phase.

    Args:
        env: DetectionEnv environment.
        replay_buffer: Replay buffer for storing transitions.
        target_update_freq: Frequency to update target network.
        criterion: Loss function.
        name: Agent name.
        network: Q-network architecture.
        exploration_mode: Exploration strategy.
    """
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ,
                 criterion=nn.SmoothL1Loss(), name="SizeDQN", network=DQN,
                 exploration_mode=EXPLORATION_MODE):
        super().__init__(env, replay_buffer, target_update_freq, criterion, name, network,
                         exploration_mode, noutputs=4)  # 4 actions: bigger, smaller, fatter, taller

    def expert_agent_action_selection(self):
        """
        Select action using expert policy based on IoU.

        Returns:
            int: Selected action.
        """
        positive_actions = []
        negative_actions = []
        old_state = self.env.bbox
        target_bboxes = self.env.current_gt_bboxes
        for action in range(4, 8):  # Actions 4-7
            new_state = self.env.transform_action(action)
            reward = self.env.calculate_reward(new_state, old_state, target_bboxes, phase='size')
            if reward > 0:
                positive_actions.append(action - 4)
            else:
                negative_actions.append(action - 4)
        return random.choice(positive_actions) if positive_actions else random.choice(negative_actions)
