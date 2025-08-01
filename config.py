# --- START OF FILE config.py ---

import torch

# General Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Replay Buffer & Batching
BUFFER_SIZE = 10000
BATCH_SIZE = 64
REPLAY_BUFFER_CAPACITY = 10000

# Imitation Learning Config
# NUM_IL_TRAJECTORIES = 1000  # Increased for better pre-training
# NUM_IL_EPOCHS = 100

NUM_IL_TRAJECTORIES = 50      # GIẢM: Chỉ tạo 10 quỹ đạo để test
NUM_IL_EPOCHS = 5           # GIẢM: Chỉ huấn luyện IL trong 5 epochs

# DQN Agent Config
TARGET_UPDATE_FREQ = 100
EXPLORATION_MODE = 'GUIDED_EXPLORE' # or 'RANDOM'
ALPHA = 0.001  # Learning rate
GAMMA = 0.99   # Discount factor

# Epsilon-Greedy Exploration
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.999
SUCCESS_CRITERIA_EPS = 1000

# Environment Config
NU = 0.5
THRESHOLD = 0.3  # IoU threshold for successful trigger
MAX_STEPS = 200
TRIGGER_STEPS = 10
NUMBER_ACTIONS = 6  # 4 move + 1 trigger + 1 confidence
ACTION_HISTORY_SIZE = 7

# Dataset and Model Config
OBJ_CONFIG = 'MULTI_OBJECT'
N_CLASSES = 20
TARGET_SIZE = (448, 448)
WINDOW_SIZE = (448, 448)
FEATURE_DIM = 512
USE_DATASET = True
# EPOCHS = 100 # Training epochs for DQN
EPOCHS = 50                 # GIẢM: Chỉ huấn luyện DQN trong 20 episodes

# Main configuration dictionary to be passed around
env_config = {
    "dataset": None,
    "current_class": None,
    "phase": 'il',  # Default phase, will be changed dynamically
    "device": device,
    
    # DQN Agent params
    "alpha": ALPHA,
    "target_update_freq": TARGET_UPDATE_FREQ,
    "exploration_mode": EXPLORATION_MODE,
    
    # Environment params
    "nu": NU,
    "threshold": THRESHOLD,
    "max_steps": MAX_STEPS,
    "trigger_steps": TRIGGER_STEPS,
    "number_actions": NUMBER_ACTIONS,
    "action_history_size": ACTION_HISTORY_SIZE,
    "object_config": OBJ_CONFIG,
    
    # Model & Data params
    "n_classes": N_CLASSES,
    "target_size": TARGET_SIZE,
    "feature_dim": FEATURE_DIM,
    "use_dataset": USE_DATASET,
    "epochs": EPOCHS,
    "window_size": WINDOW_SIZE,

    # IL params
    "num_il_trajectories": NUM_IL_TRAJECTORIES,
}
# --- END OF FILE config.py ---