import torch
import random, numpy as np
from pathlib import Path

from neural import MarioNet
from collections import deque


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 1e5  # 训练前至少需要收集的经验数量
        self.learn_every = 3   # 每隔多少条经验更新一次 Q_online
        self.sync_every = 1e4   # 每隔多少条经验同步一次 Q_target 和 Q_online

        self.save_every = 5e5   # 每隔多少条经验保存一次 MarioNet
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # Mario 用于预测最优动作的深度神经网络
        self.net = MarioNet(self.state_dim, self.action_dim).float().to(self.device)
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()


    def act(self, state):
        """
        给定一个状态，选择一个 epsilon-greedy 动作，并更新步数。

        输入：
        state(LazyFrame)：当前状态的一次观测，维度为 (state_dim)
        输出：
        action_idx (int)：表示 Mario 将执行哪个动作的整数
        """
        # 探索
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # 利用
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # 降低 exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # 递增 step
        self.curr_step += 1
        return action_idx

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
        state = np.array(state, copy=True)
        next_state = np.array(next_state, copy=True)

        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        从记忆中取回一批经验
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = zip(*batch)
        state = torch.as_tensor(np.array(state), dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.long, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        return state, next_state, action, reward, done


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


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # 从记忆中采样
        state, next_state, action, reward, done = self.recall()

        # 获取 TD 估计
        td_est = self.td_estimate(state, action)

        # 获取 TD 目标
        td_tgt = self.td_target(reward, next_state, done)

        # 通过 Q_online 反向传播损失
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet 已保存到 {save_path} 位于步数 {self.curr_step}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} 不存在")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"正在加载模型 {load_path} ，探索率为 {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
