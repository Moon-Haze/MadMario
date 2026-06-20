import random

import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def _encode_state(self, state):
        state = np.asarray(state)
        if state.dtype == np.uint8:
            return np.array(state, copy=True)
        state = np.clip(state, 0.0, 1.0)
        return np.rint(state * 255.0).astype(np.uint8)

    def push(self, state, next_state, action, reward, done):
        self.memory.append((
            self._encode_state(state),
            self._encode_state(next_state),
            int(action),
            float(reward),
            bool(done),
        ))

    def sample(self, batch_size, device):
        batch = random.sample(self.memory, batch_size)
        state, next_state, action, reward, done = zip(*batch)

        state = torch.as_tensor(np.stack(state), dtype=torch.float32, device=device).div_(255.0)
        next_state = torch.as_tensor(np.stack(next_state), dtype=torch.float32, device=device).div_(255.0)
        action = torch.as_tensor(action, dtype=torch.long, device=device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        done = torch.as_tensor(done, dtype=torch.bool, device=device)
        return state, next_state, action, reward, done
