# --- START OF FILE agents/size_agent.py ---
from .base_agent import DQNAgent
from training.policies import select_expert_action_size

class SizeDQNAgent(DQNAgent):
    def __init__(self, env, replay_buffer, **kwargs):
        super().__init__(env, replay_buffer, name="SizeDQN", **kwargs)

    def expert_agent_action_selection(self):
        """Overrides base method with specific logic for size agent."""
        target_bbox = self.env.target_bbox
        return select_expert_action_size(self.env, self.env.bbox, target_bbox)
# --- END OF FILE agents/size_agent.py ---