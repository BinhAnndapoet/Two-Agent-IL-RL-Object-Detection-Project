# --- START OF FILE utils/replay_buffer.py ---
import torch
import random
import numpy as np
from collections import namedtuple, deque
from config import REPLAY_BUFFER_CAPACITY, BATCH_SIZE

# Transition named tuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'next_state'))

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_CAPACITY, batch_size=BATCH_SIZE):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.rewards = deque(maxlen=capacity)

    def append(self, transition):
        self.memory.append(transition)
        self.rewards.append(transition.reward)

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*batch))
        
        states_list = [s.flatten() for s in batch.state]
        next_states_list = [ns.flatten() for ns in batch.next_state]
        
        max_len = max(len(s) for s in states_list)
        
        padded_states = [np.pad(s, (0, max_len - len(s)), mode='constant') for s in states_list]
        padded_next_states = [np.pad(ns, (0, max_len - len(ns)), mode='constant') for ns in next_states_list]
        
        states = torch.from_numpy(np.array(padded_states, dtype=np.float32))
        actions = torch.from_numpy(np.array(batch.action, dtype=np.int64)).unsqueeze(1)
        rewards = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).unsqueeze(1)
        dones = torch.from_numpy(np.array(batch.done, dtype=np.bool_)).unsqueeze(1).to(torch.bool)
        next_states = torch.from_numpy(np.array(padded_next_states, dtype=np.float32))
        
        if states.ndim == 2:
            states = states.unsqueeze(1)
        if next_states.ndim == 2:
            next_states = next_states.unsqueeze(1)
            
        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.memory)
# --- END OF FILE utils/replay_buffer.py ---