"""
AI 驱动的 Mario
================

作者： `Yuansong Feng <https://github.com/YuansongFeng>`__, `Suraj
Subramanian <https://github.com/suraj813>`__, `Howard
Wang <https://github.com/hw26>`__, `Steven
Guo <https://github.com/GuoYuzhang>`__.

欢迎！
--------

本教程将带你了解深度强化学习的基础知识。完成本教程后，你将实现一个
AI 驱动的 Mario（使用 `Double Deep Q-Networks <https://arxiv.org/pdf/1509.06461.pdf>`__），
让它能够自己玩游戏。

虽然本教程不要求你具备强化学习（RL）的先验知识，但你可以先熟悉这些 RL
`概念 <https://spinningup.openai.com/en/latest/spinningup/rl_intro.html>`__，
并把这份方便的
`速查表 <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N>`__
作为参考。完整代码可在
`这里 <https://github.com/yuansongFeng/MadMario/>`__ 获取。

"""


######################################################################
# 环境设置
# -----
#

# Mario 游戏环境
!pip install gymnasium "gym-super-mario-bros>=9.1.0" scikit-image

import os
import copy
import torch
from torch import nn
from pathlib import Path
from collections import deque
import random, datetime, numpy as np
from skimage import transform


# Gymnasium 是 Farama Foundation 维护的强化学习工具包（Gymnasium 的维护分支）
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStackObservation as FrameStack, GrayscaleObservation as GrayScaleObservation

# NES 模拟器包装器
from nes_py.wrappers import JoypadSpace

# Super Mario 环境
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


######################################################################
# 强化学习定义
# --------------
#
# **环境（Environment）** 智能体与之交互并从中学习的世界。
#
# **动作（Action）** :math:`a`：智能体对环境做出的响应。
# 所有可能动作的集合称为*动作空间（action-space）*。
#
# **状态（State）** :math:`s`：环境当前的特征。
# 环境所有可能状态的集合称为*状态空间（state-space）*。
#
# **奖励（Reward）** :math:`r`：奖励是环境给予智能体的关键反馈。
# 它驱动智能体学习并改变未来的动作。多个时间步上的奖励汇总称为
# **回报（Return）**。
#
# **最优动作价值函数（Optimal Action-Value function）** :math:`Q^*(s,a)`：
# 表示如果你从状态 :math:`s` 开始，采取任意动作 :math:`a`，
# 之后在每个未来时间步都采取能最大化回报的动作，那么期望回报是多少。
# 可以说 :math:`Q` 代表某个状态下动作的“质量”。我们会尝试近似这个函数。
#


######################################################################
# 初始化环境
# ======================
#
# 在 Mario 中，环境由管道、蘑菇和其他组件组成。
#
# 当 Mario 执行动作时，环境会返回变化后的（下一个）状态、奖励以及其他信息。
#

# 初始化 Super Mario 环境
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# 使用 gym-super-mario-bros 提供的复杂动作空间
env = JoypadSpace(env, COMPLEX_MOVEMENT)

env.reset()
next_state, reward, terminated, truncated, info = env.step(action=0)
done = terminated or truncated
print(f'{next_state.shape},\n {reward},\n {done},\n {info}')


######################################################################
# 预处理环境
# ======================
#
# 环境数据会通过 ``next_state`` 返回给智能体。正如你在上面看到的，
# 每个状态都由一个大小为 ``[3, 240, 256]`` 的数组表示。通常这包含了
# 比智能体所需更多的信息；例如，Mario 的动作并不取决于管道或天空的颜色！
#
# 我们使用**包装器（Wrappers）**在环境数据发送给智能体之前对其进行预处理。
#
# ``GrayScaleObservation`` 是一个常用包装器，用于将 RGB 图像转换为灰度图；
# 这样做可以在不丢失有用信息的情况下减小状态表示的大小。
# 此时每个状态的大小为：``[1, 240, 256]``
#
# ``ResizeObservation`` 会将每个观测下采样为正方形图像。
# 新的大小为：``[1, 84, 84]``
#
# ``SkipFrame`` 是一个自定义包装器，它继承自 ``gymnasium.Wrapper``，
# 并实现了 ``step()`` 函数。由于连续帧之间变化不大，我们可以跳过 n 个
# 中间帧，而不会丢失太多信息。第 n 帧会汇总每个被跳过帧中累计的奖励。
#
# ``FrameStack`` 是一个包装器，允许我们把环境中的连续帧压缩成一个单独的
# 观测点，并将其输入到学习模型中。这样，我们可以根据前几帧中的运动方向
# 判断 Mario 是在落地还是跳跃。
#

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        # 将浮点数转回 uint8
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """只返回每第 `skip` 帧"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """重复执行动作，并累加奖励"""
        total_reward = 0.0
        terminated = False
        truncated = False
        for i in range(self._skip):
            # 累加奖励，并重复执行同一个动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class NormalizeObservation(gym.ObservationWrapper):
    """将观测从 uint8 [0, 255] 归一化为 float32 [0, 1]。"""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation, dtype=np.float32) / 255.0


# 将包装器应用到环境
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = NormalizeObservation(env)
env = FrameStack(env, stack_size=4)


######################################################################
# 将上述包装器应用到环境之后，最终包装后的状态由 4 个连续的灰度帧堆叠而成，
# 如上方左侧图片所示。每当 Mario 执行动作时，环境都会返回一个具有这种结构的状态。
# 该结构由一个大小为 ``[4, 84, 84]`` 的三维数组表示。
#
# .. figure:: https://drive.google.com/uc?id=1zZU63qsuOKZIOwWt94z6cegOF2SMEmvD
#    :alt: 图片
#
#    图片
#


######################################################################
# 智能体
# =====
#
# 我们创建一个 ``Mario`` 类来表示游戏中的智能体。Mario 应该能够：
#
# - 根据当前（环境）状态，按照最优动作策略进行**行动**。
#
# - **记住**经验。经验 =（当前状态、当前动作、奖励、下一个状态）。
#   Mario 会*缓存*自己的经验，并在之后*回忆*这些经验来更新动作策略。
#
# - 随着时间推移**学习**出更好的动作策略
#

class Mario:
    def __init__():
        pass

    def act(self, state):
        """给定一个状态，选择一个 epsilon-greedy 动作"""
        pass

    def cache(self, experience):
        """将经验添加到记忆中"""
        pass

    def recall(self):
        """从记忆中采样经验"""
        pass

    def learn(self):
        """使用一批经验更新在线动作价值（Q）函数"""
        pass


######################################################################
# 在接下来的章节中，我们将填充 Mario 的参数并定义它的函数。
#


######################################################################
# 行动
# ===
#
# 对于任意给定状态，智能体可以选择执行最优动作（**利用 exploit**），
# 也可以选择随机动作（**探索 explore**）。
#
# Mario 会以 ``self.exploration_rate`` 的概率进行随机探索；当它选择利用时，
# 会依赖 ``MarioNet``（在``学习``章节中实现）来给出最优动作。
#

class Mario:
  def __init__(self, state_dim, action_dim, save_dir):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.save_dir = save_dir

    self.use_cuda = torch.cuda.is_available()

    # Mario 用于预测最优动作的深度神经网络
    self.net = MarioNet(self.state_dim, self.action_dim).float()
    if self.use_cuda:
      self.net = self.net.to(device='cuda')

    self.exploration_rate = 1
    self.exploration_rate_decay = 0.99999975
    self.exploration_rate_min = 0.1
    self.curr_step = 0

    self.save_every = 5e5   # 每隔多少条经验保存一次 MarioNet


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
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        state = state.unsqueeze(0)
        action_values = self.net(state, model='online')
        action_idx = torch.argmax(action_values, axis=1).item()

    # 降低 exploration_rate
    self.exploration_rate *= self.exploration_rate_decay
    self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

    # 递增 step
    self.curr_step += 1
    return action_idx



######################################################################
# 缓存与回忆
# ================
#
# 这两个函数构成了 Mario 的“记忆”过程。
#
# ``cache()``：每当 Mario 执行一个动作时，它都会把 ``experience`` 存入自己的记忆中。
# 它的经验包括当前*状态*、执行的*动作*、该动作带来的*奖励*、*下一个状态*，
# 以及游戏是否已经*结束*。
#
# ``recall()``：Mario 会从记忆中随机采样一批经验，并利用这些经验来学习游戏。
#

class Mario(Mario): # 通过继承保持教程代码的连续性
  def __init__(self, state_dim, action_dim, save_dir):
    super().__init__(state_dim, action_dim, save_dir)
    self.memory = deque(maxlen=100000)
    self.batch_size = 32


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
    state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
    action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
    reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
    done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

    self.memory.append( (state, next_state, action, reward, done,) )


  def recall(self):
    """
    从记忆中取回一批经验
    """
    batch = random.sample(self.memory, self.batch_size)
    state, next_state, action, reward, done = map(torch.stack, zip(*batch))
    return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


######################################################################
# 学习
# =====
#
# Mario 底层使用 `DDQN 算法 <https://arxiv.org/pdf/1509.06461>`__。
# DDQN 使用两个卷积网络——:math:`Q_{online}` 和 :math:`Q_{target}`——
# 它们各自独立地近似最优动作价值函数。
#
# 在我们的实现中，:math:`Q_{online}` 和 :math:`Q_{target}` 共享特征生成器
# ``features``，但分别维护独立的全连接分类器。
# :math:`\theta_{target}`（:math:`Q_{target}` 的参数）被冻结，以防止通过
# 反向传播更新。相反，它会周期性地与 :math:`\theta_{online}` 同步（稍后会详细说明）。
#


######################################################################
# 神经网络
# ~~~~~~~~~~~~~~
#

class MarioNet(nn.Module):
  '''迷你 CNN 结构
  输入 -> (conv2d + relu) x 3 -> 展平 -> (dense + relu) x 2 -> 输出
  '''
  def __init__(self, input_dim, output_dim):
      super().__init__()
      c, h, w = input_dim

      if h != 84:
          raise ValueError(f"期望输入高度为: 84, 实际为: {h}")
      if w != 84:
          raise ValueError(f"期望输入宽度为: 84, 实际为: {w}")

      self.online = nn.Sequential(
          nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(3136, 512),
          nn.ReLU(),
          nn.Linear(512, output_dim)
      )

      self.target = copy.deepcopy(self.online)

      # 冻结 Q_target 的参数。
      for p in self.target.parameters():
          p.requires_grad = False

  def forward(self, input, model):
      if model == 'online':
          return self.online(input)
      elif model == 'target':
          return self.target(input)


######################################################################
# TD 估计与 TD 目标
# -----------------------
#
# 学习过程中涉及两个值：
#
# **TD 估计（TD Estimate）**——给定状态 :math:`s` 下预测的最优 :math:`Q^*`
#
# .. math::
#
#
#    {TD}_e = Q_{online}^*(s,a)
#
# **TD 目标（TD Target）**——当前奖励与下一个状态 :math:`s'` 中估计的
# :math:`Q^*` 的聚合
#
# .. math::
#
#
#    a' = argmax_{a} Q_{online}(s', a)
#
# .. math::
#
#
#    {TD}_t = r + \gamma Q_{target}^*(s',a')
#
# 因为我们不知道下一个动作 :math:`a'` 会是什么，所以使用在下一个状态
# :math:`s'` 中能使 :math:`Q_{online}` 最大的动作 :math:`a'`。
#
# 注意，我们在 ``td_target()`` 上使用了
# [@torch.no_grad()](https://pytorch.org/docs/stable/generated/torch.no_grad.html#no-grad)
# 装饰器，以在这里禁用梯度计算（因为我们不需要对 :math:`\theta_{target}` 进行反向传播）。
#

class Mario(Mario):
  def __init__(self, state_dim, action_dim, save_dir):
    super().__init__(state_dim, action_dim, save_dir)
    self.gamma = 0.9

  def td_estimate(self, state, action):
    current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
    return current_Q

  @torch.no_grad()
  def td_target(self, reward, next_state, done):
    next_state_Q = self.net(next_state, model='online')
    best_action = torch.argmax(next_state_Q, axis=1)
    next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
    return (reward + (1 - done.float()) * self.gamma * next_Q).float()


######################################################################
# 更新模型
# ------------------
#
# 当 Mario 从回放缓冲区中采样输入时，我们计算 :math:`TD_t` 和 :math:`TD_e`，
# 并将该损失通过 :math:`Q_{online}` 反向传播，以更新其参数
# :math:`\theta_{online}`（:math:`\alpha` 是传给 ``Adam optimizer`` 的学习率 ``lr``）
#
# .. math::
#
#
#    \theta_{online} \leftarrow \theta_{online} + \alpha \nabla(TD_e - TD_t)
#
# :math:`\theta_{target}` 不会通过反向传播更新。
# 相反，我们会周期性地将 :math:`\theta_{online}` 复制到 :math:`\theta_{target}`
#
# .. math::
#
#
#    \theta_{target} \leftarrow \theta_{online}
#
#

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
      super().__init__(state_dim, action_dim, save_dir)
      self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
      self.loss_fn = torch.nn.SmoothL1Loss()

    def update_Q_online(self, td_estimate, td_target) :
      loss = self.loss_fn(td_estimate, td_target)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      return loss.item()

    def sync_Q_target(self):
      self.net.target.load_state_dict(self.net.online.state_dict())


######################################################################
# 保存检查点
# ---------------
#

class Mario(Mario):
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


######################################################################
# 整合起来
# -----------------------
#

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.burnin = 1e5  # 训练前至少需要收集的经验数量
        self.learn_every = 3   # 每隔多少条经验更新一次 Q_online
        self.sync_every = 1e4   # 每隔多少条经验同步一次 Q_target 和 Q_online


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



######################################################################
# 日志记录
# =======
#

import numpy as np
import time, datetime
import matplotlib.pyplot as plt

class MetricLogger():
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'回合':>8}{'步数':>8}{'探索率':>10}{'平均奖励':>15}"
                f"{'平均长度':>15}{'平均损失':>15}{'平均Q值':>15}"
                f"{'时间间隔':>15}{'时间':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # 历史指标
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # 移动平均值，每次调用 record() 时都会追加
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # 当前回合指标
        self.init_episode()

        # 计时
        self.record_time = time.time()


    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "记录一个回合的结束"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)


        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"回合 {episode} - "
            f"步数 {step} - "
            f"Epsilon {epsilon} - "
            f"平均奖励 {mean_ep_reward} - "
            f"平均长度 {mean_ep_length} - "
            f"平均损失 {mean_ep_loss} - "
            f"平均 Q 值 {mean_ep_q} - "
            f"时间间隔 {time_since_last_record} - "
            f"时间 {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()



######################################################################
# 开始玩吧！
# ===========
#

use_cuda = torch.cuda.is_available()
print(f"使用 CUDA： {use_cuda}")
print()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 40000

### 通过玩游戏的方式，循环训练模型 num_episodes 次
for e in range(episodes):

    state, _ = env.reset()

    # 开始游戏！
    while True:

        # 让智能体基于当前状态运行
        action = mario.act(state)

        # 智能体执行动作
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 记住经验
        mario.cache(state, next_state, action, reward, done)

        # 学习
        q, loss = mario.learn()

        # 日志记录
        logger.log_step(reward, loss, q)

        # 更新状态
        state = next_state

        # 检查游戏是否结束
        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
