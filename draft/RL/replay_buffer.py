from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, action, reward, next_states, dones = map(np.array, zip(*batch))
        return states, action, reward, next_states, dones
    
    def __len__(self):
        return len(self.buffer)