import torch
import numpy as np
import os
from env import DetectionEnv
from agents import CenterDQNAgent, SizeDQNAgent
from utils import ReplayBuffer
from IL import generate_expert_trajectory, train_il_model, initialize_replay_buffer

def train_imitation_learning(env):
    """
    Train IL models for Center and Size agents.

    Args:
        env (DetectionEnv): Environment instance.

    Returns:
        tuple: (center_il_model, size_il_model)
    """
    print("\033[92mGenerating expert trajectories for IL...\033[0m")
    trajectories = generate_expert_trajectory(env, num_trajectories=NUM_IL_TRAJECTORIES)
    
    print("\033[92mTraining IL model for CenterDQNAgent...\033[0m")
    center_il_model = train_il_model(env, trajectories['center'], phase='center', epochs=NUM_IL_EPOCHS)
    
    print("\033[92mTraining IL model for SizeDQNAgent...\033[0m")
    size_il_model = train_il_model(env, trajectories['size'], phase='size', epochs=NUM_IL_EPOCHS)
    
    return center_il_model, size_il_model


def initialize_agents_and_buffer(env, center_il_model, size_il_model):
    """
    Initialize agents and replay buffer with IL transitions.

    Args:
        env (DetectionEnv): Environment instance.
        center_il_model (ILModel): IL model for center phase.
        size_il_model (ILModel): IL model for size phase.

    Returns:
        tuple: (center_agent, size_agent, replay_buffer)
    """
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)
    
    print("\033[92mInitializing replay buffer with IL transitions...\033[0m")
    initialize_replay_buffer(env, center_il_model, size_il_model, replay_buffer, num_transitions=NUM_IL_TRAJECTORIES)
    
    center_agent = CenterDQNAgent(env, replay_buffer)
    size_agent = SizeDQNAgent(env, replay_buffer)
    
    return center_agent, size_agent, replay_buffer
