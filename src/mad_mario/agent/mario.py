import numpy as np
import torch

from mad_mario.agent.replay_buffer import ReplayBuffer
from mad_mario.config import AgentConfig, TrainingConfig
from mad_mario.models.mario_net import MarioNet


class Mario:
    def __init__(self, state_dim, action_dim, agent_config: AgentConfig, training_config: TrainingConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = training_config.batch_size
        self.memory = ReplayBuffer(agent_config.replay_capacity)

        self.exploration_rate = agent_config.exploration_rate
        self.exploration_rate_decay = agent_config.exploration_rate_decay
        self.exploration_rate_min = agent_config.exploration_rate_min
        self.gamma = agent_config.gamma

        self.curr_step = 0
        self.current_episode = 0
        self.loaded_episode = 0
        self.burnin = training_config.burnin
        self.learn_every = training_config.learn_every
        self.sync_every = training_config.sync_every
        self.next_learn_step = self._next_interval_step(self.burnin, self.learn_every)
        self.next_sync_step = self.sync_every

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.net = MarioNet(self.state_dim, self.action_dim).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=agent_config.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        """给定一个状态，选择一个 epsilon-greedy 动作，并更新步数。"""
        return self.act_batch(np.expand_dims(state, axis=0))[0]

    def act_batch(self, states):
        """给定一批状态，为每个环境选择一个 epsilon-greedy 动作。"""
        states = np.asarray(states)
        batch_size = len(states)
        actions = np.empty(batch_size, dtype=np.int64)
        explore_mask = np.random.rand(batch_size) < self.exploration_rate

        actions[explore_mask] = np.random.randint(self.action_dim, size=explore_mask.sum())
        exploit_indices = np.flatnonzero(~explore_mask)
        if len(exploit_indices) > 0:
            exploit_states = torch.as_tensor(
                states[exploit_indices],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                action_values = self.net(exploit_states, model="online")
            actions[exploit_indices] = torch.argmax(action_values, axis=1).cpu().numpy()

        self.exploration_rate *= self.exploration_rate_decay ** batch_size
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += batch_size
        return actions.tolist()

    def cache(self, state, next_state, action, reward, done):
        self.memory.push(state, next_state, action, reward, done)

    def cache_batch(self, states, next_states, actions, rewards, dones):
        """将一批并行环境的经验存入经验回放缓冲区。"""
        for state, next_state, action, reward, done in zip(
            states,
            next_states,
            actions,
            rewards,
            dones,
        ):
            self.cache(state, next_state, action, reward, done)

    def recall(self):
        """从记忆中取回一批经验。"""
        return self.memory.sample(self.batch_size, self.device)

    def td_estimate(self, state, action):
        return self.net(state, model="online")[np.arange(0, self.batch_size), action]

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_q, axis=1)
        next_q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def _next_interval_step(self, current_step, interval):
        return ((int(current_step) // interval) + 1) * interval

    def _learn_once(self):
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss

    def learn(self):
        q, loss = None, None

        while self.curr_step >= self.next_sync_step:
            self.sync_Q_target()
            self.next_sync_step += self.sync_every

        if self.curr_step < self.burnin or len(self.memory) < self.batch_size:
            return None, None

        while self.curr_step >= self.next_learn_step:
            q, loss = self._learn_once()
            self.next_learn_step += self.learn_every

        return q, loss

    def state_dict(self):
        return {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "exploration_rate": self.exploration_rate,
            "curr_step": self.curr_step,
            "episode": self.current_episode,
        }

    def load_state_dict(self, checkpoint):
        exploration_rate = checkpoint.get("exploration_rate", self.exploration_rate)
        state_dict = checkpoint.get("model")
        optimizer_state = checkpoint.get("optimizer")
        curr_step = checkpoint.get("curr_step", 0)
        episode = checkpoint.get("episode", 0)

        self.net.load_state_dict(state_dict)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.exploration_rate = exploration_rate
        self.curr_step = curr_step
        self.current_episode = episode
        self.loaded_episode = episode
        self.next_learn_step = self._next_interval_step(max(self.curr_step, self.burnin), self.learn_every)
        self.next_sync_step = self._next_interval_step(self.curr_step, self.sync_every)
