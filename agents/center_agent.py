# --- START OF FILE agents/center_agent.py ---
from .base_agent import DQNAgent
from training.policies import select_expert_action_center

class CenterDQNAgent(DQNAgent):
    def __init__(self, env, replay_buffer, **kwargs):
        super().__init__(env, replay_buffer, name="CenterDQN", **kwargs)

    def expert_agent_action_selection(self):
        """Overrides base method with specific logic for center agent."""
        target_bbox = self.env.target_bbox
        target_label = self.env.current_gt_labels[self.env.current_gt_index]
        pos_action, class_action, _, _ = select_expert_action_center(self.env, self.env.bbox, target_bbox, target_label)
        return (pos_action, class_action)
# --- END OF FILE agents/center_agent.py ---