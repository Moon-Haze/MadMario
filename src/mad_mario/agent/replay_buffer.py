import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, state_dim, alpha=0.6, epsilon=1e-6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        self.states = [None] * self.capacity
        self.next_states = [None] * self.capacity
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=np.bool_)
        self.n_steps = np.empty(self.capacity, dtype=np.float32)
        self.priorities = np.zeros(self.capacity, dtype=np.float32)

    def __len__(self):
        return self.size

    def _encode_state(self, state):
        state = np.asarray(state)
        if state.dtype == np.uint8:
            return np.array(state, copy=True)
        state = np.clip(state, 0.0, 1.0)
        return np.rint(state * 255.0).astype(np.uint8)

    def push(self, state, next_state, action, reward, done, n_step=1):
        self.states[self.pos] = self._encode_state(state)
        self.next_states[self.pos] = self._encode_state(next_state)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = bool(done)
        self.n_steps[self.pos] = float(n_step)
        self.priorities[self.pos] = self.max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta, device):
        priorities = self.priorities[: self.size]
        scaled_priorities = priorities ** self.alpha
        priority_sum = scaled_priorities.sum()
        if not np.isfinite(priority_sum) or priority_sum <= 0:
            probabilities = np.full(self.size, 1.0 / self.size, dtype=np.float64)
        else:
            probabilities = (scaled_priorities / priority_sum).astype(np.float64)
            probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, size=batch_size, replace=True, p=probabilities)
        sample_probabilities = probabilities[indices]
        weights = (self.size * sample_probabilities) ** (-float(beta))
        weights /= weights.max()

        state = torch.as_tensor(np.stack([self.states[index] for index in indices]), dtype=torch.float32, device=device).div_(255.0)
        next_state = torch.as_tensor(
            np.stack([self.next_states[index] for index in indices]),
            dtype=torch.float32,
            device=device,
        ).div_(255.0)
        action = torch.as_tensor(self.actions[indices], dtype=torch.long, device=device)
        reward = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device)
        done = torch.as_tensor(self.dones[indices], dtype=torch.bool, device=device)
        n_step = torch.as_tensor(self.n_steps[indices], dtype=torch.float32, device=device)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        indices = torch.as_tensor(indices, dtype=torch.long, device=device)
        return state, next_state, action, reward, done, n_step, weights, indices

    def update_priorities(self, indices, td_errors):
        indices = indices.detach().cpu().numpy()
        priorities = np.abs(td_errors.detach().cpu().numpy()) + self.epsilon
        self.priorities[indices] = priorities.astype(np.float32)
        self.max_priority = max(self.max_priority, float(priorities.max()))
