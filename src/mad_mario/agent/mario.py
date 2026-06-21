import math
from collections import defaultdict, deque

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
        self.memory = ReplayBuffer(
            agent_config.replay_capacity,
            state_dim,
            alpha=agent_config.per_alpha,
            epsilon=agent_config.per_epsilon,
        )

        self._noisy = agent_config.noisy_std_init > 0
        self._noisy_std_init = float(agent_config.noisy_std_init)
        self.exploration_rate_start = agent_config.exploration_rate
        self.exploration_rate = agent_config.exploration_rate
        self.exploration_decay_steps = agent_config.exploration_decay_steps
        self.exploration_rate_min = agent_config.exploration_rate_min
        self.gradient_clip_norm = agent_config.gradient_clip_norm
        self.gamma = agent_config.gamma
        self.n_step = max(1, int(agent_config.n_step))
        self.per_beta_start = agent_config.per_beta_start
        self.per_beta_frames = agent_config.per_beta_frames

        self._lr_warmup_steps = int(agent_config.lr_warmup_steps)
        self._lr_min_ratio = float(agent_config.lr_min_ratio)
        self._lr_cosine_steps = max(self._lr_warmup_steps + 1, int(agent_config.exploration_decay_steps))

        self.curr_step = 0
        self.current_episode = 0
        self.loaded_episode = 0
        self.burnin = training_config.burnin
        self.learn_every = training_config.learn_every
        self.sync_every = training_config.sync_every
        self.next_learn_step = self._next_interval_step(self.burnin, self.learn_every)
        self.next_sync_step = self.sync_every
        self.n_step_buffers = defaultdict(deque)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.net = MarioNet(self.state_dim, self.action_dim, noisy_std_init=agent_config.noisy_std_init).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=agent_config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda,
        )
        self.loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    def act(self, state):
        """给定一个状态，选择一个 epsilon-greedy 动作，并更新步数。"""
        return self.act_batch(np.expand_dims(state, axis=0))[0]

    def act_batch(self, states):
        """给定一批状态，为每个环境选择一个动作。

        使用 NoisyNet 时直接 argmax（网络内噪声提供探索）；
        否则使用 epsilon-greedy。
        """
        states = np.asarray(states)
        batch_size = len(states)
        states_tensor = self._state_tensor(states)

        if self._noisy:
            self.net.online.train()
            with torch.no_grad():
                action_values = self.net(states_tensor, model="online")
            actions = torch.argmax(action_values, axis=1).cpu().numpy()
        else:
            actions = np.empty(batch_size, dtype=np.int64)
            explore_mask = np.random.rand(batch_size) < self.exploration_rate
            actions[explore_mask] = np.random.randint(self.action_dim, size=explore_mask.sum())
            exploit_indices = np.flatnonzero(~explore_mask)
            if len(exploit_indices) > 0:
                exploit_states = states_tensor[exploit_indices]
                with torch.no_grad():
                    action_values = self.net(exploit_states, model="online")
                actions[exploit_indices] = torch.argmax(action_values, axis=1).cpu().numpy()

        self.curr_step += batch_size
        if not self._noisy:
            self._update_exploration_rate()
        return actions.tolist()

    def act_eval(self, state):
        return self.act_eval_batch(np.expand_dims(state, axis=0))[0]

    def act_eval_batch(self, states):
        states = self._state_tensor(np.asarray(states))
        self.net.online.eval()
        with torch.no_grad():
            action_values = self.net(states, model="online")
        return torch.argmax(action_values, axis=1).cpu().numpy().tolist()

    def _state_tensor(self, states):
        states_array = np.asarray(states)
        states_tensor = torch.as_tensor(states_array, dtype=torch.float32, device=self.device)
        if states_array.dtype == np.uint8:
            states_tensor = states_tensor.div(255.0)
        return states_tensor

    def _update_exploration_rate(self):
        if self.exploration_decay_steps <= 0:
            self.exploration_rate = self.exploration_rate_min
            return
        progress = min(1.0, self.curr_step / self.exploration_decay_steps)
        self.exploration_rate = self.exploration_rate_start + progress * (
            self.exploration_rate_min - self.exploration_rate_start
        )
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

    def _lr_lambda(self, step):
        if step < self._lr_warmup_steps:
            return max(self._lr_min_ratio, float(step) / self._lr_warmup_steps)
        if step >= self._lr_cosine_steps:
            return self._lr_min_ratio
        progress = (step - self._lr_warmup_steps) / (self._lr_cosine_steps - self._lr_warmup_steps)
        return self._lr_min_ratio + 0.5 * (1.0 - self._lr_min_ratio) * (1.0 + math.cos(math.pi * progress))

    def cache(self, state, next_state, action, reward, done):
        self._cache_transition(0, state, next_state, action, reward, done)

    def cache_batch(self, states, next_states, actions, rewards, dones):
        """将一批并行环境的经验存入经验回放缓冲区。"""
        for env_index, (state, next_state, action, reward, done) in enumerate(zip(
            states,
            next_states,
            actions,
            rewards,
            dones,
        )):
            self._cache_transition(env_index, state, next_state, action, reward, done)

    def _cache_transition(self, env_index, state, next_state, action, reward, done):
        buffer = self.n_step_buffers[env_index]
        buffer.append((state, next_state, action, reward, done))

        if len(buffer) >= self.n_step:
            self._push_n_step_transition(buffer)
            buffer.popleft()

        if done:
            while buffer:
                self._push_n_step_transition(buffer)
                buffer.popleft()

    def _push_n_step_transition(self, buffer):
        state, _, action, _, _ = buffer[0]
        reward = 0.0
        next_state = buffer[-1][1]
        done = False
        n_steps = 0
        for step, (_, transition_next_state, _, transition_reward, transition_done) in enumerate(buffer):
            reward += (self.gamma ** step) * float(transition_reward)
            next_state = transition_next_state
            done = bool(transition_done)
            n_steps = step + 1
            if done:
                break
        self.memory.push(state, next_state, action, reward, done, n_steps)

    def recall(self):
        """从记忆中取回一批经验。"""
        return self.memory.sample(self.batch_size, self._per_beta(), self.device)

    def _per_beta(self):
        progress = min(1.0, self.curr_step / max(1, self.per_beta_frames))
        return self.per_beta_start + progress * (1.0 - self.per_beta_start)

    def td_estimate(self, state, action):
        batch_indices = torch.arange(action.shape[0], device=self.device)
        return self.net(state, model="online")[batch_indices, action]

    @torch.no_grad()
    def td_target(self, reward, next_state, done, n_steps):
        next_state_q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_q, axis=1)
        batch_indices = torch.arange(best_action.shape[0], device=self.device)
        next_q = self.net(next_state, model="target")[batch_indices, best_action]
        discount = torch.pow(torch.full_like(n_steps, self.gamma), n_steps)
        return (reward + (1 - done.float()) * discount * next_q).float()

    def update_Q_online(self, td_estimate, td_target, weights):
        td_errors = td_target.detach() - td_estimate
        losses = self.loss_fn(td_estimate, td_target)
        loss = (losses * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.online.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item(), td_errors.detach()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def _next_interval_step(self, current_step, interval):
        return ((int(current_step) // interval) + 1) * interval

    def _learn_once(self):
        state, next_state, action, reward, done, n_steps, weights, indices = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done, n_steps)
        loss, td_errors = self.update_Q_online(td_est, td_tgt, weights)
        self.memory.update_priorities(indices, td_errors)
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
            "scheduler": self.scheduler.state_dict(),
            "exploration_rate": self.exploration_rate,
            "curr_step": self.curr_step,
            "episode": self.current_episode,
        }

    def _convert_old_state_dict(self, state_dict, noisy_std_init):
        """从 DuelingDQN (nn.Linear) 格式转换到 NoisyDuelingDQN (NoisyLinear) 格式。

        只转换 value/advantage 分支中的线性层；卷积层（features）保持不动。
        """
        has_old_format = any("value" in k or "advantage" in k for k in state_dict
                             if k.endswith(".weight") and not k.endswith("weight_mu"))
        if not has_old_format:
            return state_dict

        new_sd = {}
        for key, param in state_dict.items():
            is_value_or_adv = "value" in key or "advantage" in key

            if is_value_or_adv and key.endswith(".weight"):
                base = key.rsplit(".", 1)[0]
                new_sd[base + ".weight_mu"] = param
                new_sd[base + ".weight_sigma"] = torch.full_like(
                    param, noisy_std_init / math.sqrt(param.shape[1])
                )
            elif is_value_or_adv and key.endswith(".bias"):
                base = key.rsplit(".", 1)[0]
                new_sd[base + ".bias_mu"] = param
                new_sd[base + ".bias_sigma"] = torch.full_like(
                    param, noisy_std_init / math.sqrt(param.shape[0])
                )
            else:
                new_sd[key] = param
        return new_sd

    def load_state_dict(self, checkpoint):
        exploration_rate = checkpoint.get("exploration_rate", self.exploration_rate)
        state_dict = checkpoint.get("model")
        optimizer_state = checkpoint.get("optimizer")
        scheduler_state = checkpoint.get("scheduler")
        curr_step = checkpoint.get("curr_step", 0)
        episode = checkpoint.get("episode", 0)

        if self._noisy:
            state_dict = self._convert_old_state_dict(state_dict, self._noisy_std_init)
        self.net.load_state_dict(state_dict)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        self.exploration_rate = exploration_rate
        self.curr_step = curr_step
        self.current_episode = episode
        self.loaded_episode = episode
        self.next_learn_step = self._next_interval_step(max(self.curr_step, self.burnin), self.learn_every)
        self.next_sync_step = self._next_interval_step(self.curr_step, self.sync_every)
