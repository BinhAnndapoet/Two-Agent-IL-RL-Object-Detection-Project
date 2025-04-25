from models import *
from utils import *

import torch
import torch.nn as nn
import random

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
    def __init__(self, env, replay_buffer, target_update_freq=TARGET_UPDATE_FREQ, exploration_mode=EXPLORATION_MODE, network=DQN):
      self.env = env
      self.replay_buffer = replay_buffer
      self.target_update_freq = target_update_freq
      self.exploration_mode = exploration_mode
      self.ninputs = 
      self.noutputs = 
      self.policy_net = network(self.ninputs, self.noutputs).to(device)
      self.target_net = network(self.ninputs, self.noutputs).to(device)
      self.target_net.load_state_dict(self.policy_net.state_dict())
      self.target_net.eval()
      self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr= ALPHA)
      self.criterion = nn.SmoothL1Loss()
      self.epsilon = EPS_START
      self.steps_done = 0
      self.episodes = 0
      self.episode_info = 
      self.display_every_n_episodes = 10000
      
    def select_action(self, state, agent_type="center"):
        """ Selects an action using an epsilon greedy policy """
        # Selecting a random action with probability epsilon
        if random.random() <= self.epsilon: # Exploration
            if self.exploration_mode == GUIDED_EXPLORE: # Guided exploration
                # Expert agent action selection
                action = self.expert_agent_action_selection(state, agent_type)
            else:# Random exploration
                # Normal Random action Selection
                action = self.env.action_space.sample() if agent_type == "center" else self.env.action_space_size.sample()
        else: # Exploitation
            # Selecting the action with the highest Q-value otherwise
            with torch.no_grad():
                if agent_type == "center":
                    state = torch.cat([state["image"], state["features"]], dim=1).to(device)
                else:
                    state = torch.cat([state["image_patch"], state["features_patch"]], dim=1).to(device)
                qvalues = self.policy_net(state)
                action = qvalues.argmax().item()
        return action
    
    def expert_agent_action_selection(self):
        """ Selects an action using an expert agent, by calculating the reward for each action and selecting a random action from the positive actions if the list is not empty, otherwise selecting a random action from the negative actions.

            Returns:
                action: The action selected by the expert agent
        """
      pass
        
    def update(self):
        """ Updates the policy network using a batch of transitions """
        # Sampling a batch of transitions from the replay buffer
        states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch()  # sample.batch() from utils

        # Converting the tensors to cuda tensors
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        next_states = next_states.to(device)

            
    def update_epsilon(self):
        """ Updates epsilon """
        self.epsilon = max(EPS_END, EPS_DECAY * self.epsilon)

    def train(self):
        """ Trains the agent for nsteps steps """
        pass

    def explicit_train(self, decode=False):
        """ Trains the agent for nsteps steps """
      pass

    def run(self):
        """ Runs the agent """
        # Initialising the replay buffer
        self.replay_buffer.initialize()

        # Training the agent
        self.train()

    def evaluate(self, path="evaluation_results"):
        """ Evaluates the agent """
      pass

    def test(self, file_path='dqn_render', video_filename='output_video.mp4'):
        """ Tests the trained agent and creates an MP4 video """
      pass

    def save_gif(self, file_path='dqn_render', gif_filename='output.gif'):
        """Tests the trained agent and creates a GIF animation."""
      pass

    def save(self, path="models/dqn"):
        """ Function to save the model 
            
            Args:
                path (str): The path to save the model to
        """
        pass:

    def load(self, path="models/dqn"):
        """ Function to load the model 
            
            Args:
                path (str): The path to load the model from
        """
      pass:

    def get_episode_info(self):
        """ Returns the episode info """
      pass


class CenterAgent(DQNAgent):
    """ The Center Agent that interacts with the environment and inherits from the DQN agent """
  def __init__():
    pass
    
  def update(self):
    """ Updates the policy network using a batch of transitions """
      pass


class SizeAgent(DQNAgent):
    """ The Size Agent that interacts with the environment and inherits from the DQN agent """
  def __init__():
    pass
    
  def update(self):
    """ Updates the policy network using a batch of transitions """
      pass
    
