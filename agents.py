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
    def __init__(self):
      pass
      
    def select_action(self):
        """ Selects an action using an epsilon greedy policy """
      pass
    
    def expert_agent_action_selection(self):
        """ Selects an action using an expert agent, by calculating the reward for each action and selecting a random action from the positive actions if the list is not empty, otherwise selecting a random action from the negative actions.

            Returns:
                action: The action selected by the expert agent
        """
      pass
        
    def update(self):
        """ Updates the policy network using a batch of transitions """
      pass
            
    def update_epsilon(self):
        """ Updates epsilon """
      pass

    def train(self):
        """ Trains the agent for nsteps steps """
      pass

    def explicit_train(self, decode=False):
        """ Trains the agent for nsteps steps """
      pass

    def run(self):
        """ Runs the agent """
      pass

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
        # Creating the directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Saving the model
        torch.save(self.policy_net.state_dict(), path + "/policy_net.pth")
        torch.save(self.target_net.state_dict(), path + "/target_net.pth")

        # Saving optimizer state
        torch.save(self.optimizer.state_dict(), path + "/optimizer.pth")

        # Saving the episode info
        np.save(path + "/episode_info.npy", self.episode_info)

    def load(self, path="models/dqn"):
        """ Function to load the model 
            
            Args:
                path (str): The path to load the model from
        """
      pass:

    def get_episode_info(self):
        """ Returns the episode info """
      pass
