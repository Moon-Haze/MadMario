import torch
import numpy as np

from neural import MarioNet
from replay_buffer import ReplayBuffer
from tqdm import tqdm


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = config.batch_size if config else 32
        replay_capacity = config.replay_capacity if config else 100000
        self.memory = ReplayBuffer(replay_capacity)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.current_episode = 0
        self.loaded_episode = 0
        self.burnin = config.burnin if config else int(1e5)  # 训练前至少需要收集的经验数量
        self.learn_every = config.learn_every if config else 3   # 每隔多少条经验更新一次 Q_online
        self.sync_every = config.sync_every if config else int(1e4)   # 每隔多少条经验同步一次 Q_target 和 Q_online

        self.save_every = config.save_every if config else int(5e5)   # 每隔多少条经验保存一次 MarioNet
        self.save_dir = save_dir
        self.next_learn_step = self._next_interval_step(self.burnin, self.learn_every)
        self.next_sync_step = self.sync_every
        self.next_save_step = self.save_every

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # Mario 用于预测最优动作的深度神经网络
        self.net = MarioNet(self.state_dim, self.action_dim).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        if checkpoint:
            self.load(checkpoint)


    def act(self, state):
        """
        给定一个状态，选择一个 epsilon-greedy 动作，并更新步数。

        输入：
        state(LazyFrame)：当前状态的一次观测，维度为 (state_dim)
        输出：
        action_idx (int)：表示 Mario 将执行哪个动作的整数
        """
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
                action_values = self.net(exploit_states, model='online')
            actions[exploit_indices] = torch.argmax(action_values, axis=1).cpu().numpy()

        self.exploration_rate *= self.exploration_rate_decay ** batch_size
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += batch_size
        return actions.tolist()

    def cache(self, state, next_state, action, reward, done):
        """
        将经验存储到 self.memory（经验回放缓冲区）中

        输入：
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
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
        """
        从记忆中取回一批经验
        """
        return self.memory.sample(self.batch_size, self.device)


    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target) :
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
        # 从记忆中采样
        state, next_state, action, reward, done = self.recall()

        # 获取 TD 估计
        td_est = self.td_estimate(state, action)

        # 获取 TD 目标
        td_tgt = self.td_target(reward, next_state, done)

        # 通过 Q_online 反向传播损失
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def learn(self):
        q, loss = None, None

        while self.curr_step >= self.next_sync_step:
            self.sync_Q_target()
            self.next_sync_step += self.sync_every

        while self.curr_step >= self.next_save_step:
            self.save()
            self.next_save_step += self.save_every

        if self.curr_step < self.burnin or len(self.memory) < self.batch_size:
            return None, None

        while self.curr_step >= self.next_learn_step:
            q, loss = self._learn_once()
            self.next_learn_step += self.learn_every

        return q, loss


    def save(self, episode=None):
        if episode is not None:
            self.current_episode = episode

        checkpoint = dict(
            model=self.net.state_dict(),
            optimizer=self.optimizer.state_dict(),
            exploration_rate=self.exploration_rate,
            curr_step=self.curr_step,
            episode=self.current_episode,
            save_dir=str(self.save_dir),
        )
        save_path = self.save_dir / f"mario_net_step_{self.curr_step}_ep_{self.current_episode}.chkpt"
        latest_path = self.save_dir.parent / "latest.chkpt"
        latest_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, save_path)
        torch.save(checkpoint, latest_path)
        tqdm.write(f"MarioNet 已保存到 {save_path}，latest 已更新，当前步数 {self.curr_step}，当前回合 {self.current_episode}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} 不存在")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate', self.exploration_rate)
        state_dict = ckp.get('model')
        optimizer_state = ckp.get('optimizer')
        curr_step = ckp.get('curr_step', 0)
        episode = ckp.get('episode', 0)
        tqdm.write(f"正在加载模型 {load_path}，探索率为 {exploration_rate:.4f}，步数为 {curr_step}，回合为 {episode}")
        self.net.load_state_dict(state_dict)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.exploration_rate = exploration_rate
        self.curr_step = curr_step
        self.current_episode = episode
        self.loaded_episode = episode
        self.next_learn_step = self._next_interval_step(max(self.curr_step, self.burnin), self.learn_every)
        self.next_sync_step = self._next_interval_step(self.curr_step, self.sync_every)
        self.next_save_step = self._next_interval_step(self.curr_step, self.save_every)
