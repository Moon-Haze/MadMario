import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path

from tqdm import tqdm
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gymnasium.wrappers import FrameStackObservation as FrameStack, GrayscaleObservation as GrayScaleObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame, NormalizeObservation

# 初始化超级马里奥游戏环境，这里选择 1-1 关卡
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# 使用更复杂的动作空间重新训练更强的 Mario。
# 注意：动作数量变化后，旧的 2 动作检查点不能直接加载。
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# 对环境观测做预处理：跳帧、灰度化、缩放、归一化，并堆叠最近 4 帧作为状态
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = NormalizeObservation(env)
env = FrameStack(env, stack_size=4)

env.reset()

save_dir = Path('./checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

# 如果想从已有模型继续训练，可以把检查点改成对应的 .chkpt 文件路径
checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
# 创建智能体：输入状态为 4 帧 84x84 图像，输出动作为当前动作空间大小
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 40000

### 通过玩游戏的方式，循环训练模型 num_episodes 次
progress_bar = tqdm(range(episodes), desc="训练进度", unit="回合")
for e in progress_bar:

    state, _ = env.reset()
    ep_reward = 0.0
    ep_length = 0
    last_loss = None
    last_q = None

    # 开始游戏！
    while True:
        # 3. 显示环境（画面）
        # env.render()
        # 4. 让智能体基于当前状态运行
        action = mario.act(state)
        # 5. 智能体执行动作
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # 6. 记住经验
        mario.cache(state, next_state, action, reward, done)
        # 7. 学习
        q, loss = mario.learn()
        # 8. 日志记录
        logger.log_step(reward, loss, q)
        ep_reward += reward
        ep_length += 1
        if loss is not None:
            last_loss = loss
            last_q = q
        # 9. 更新状态
        state = next_state
        # 10. 检查游戏是否结束
        if done or info['flag_get']:
            break

    logger.log_episode()
    progress_bar.set_postfix({
        "步数": mario.curr_step,
        "探索率": f"{mario.exploration_rate:.3f}",
        "回合奖励": f"{ep_reward:.1f}",
        "回合长度": ep_length,
        "损失": "-" if last_loss is None else f"{last_loss:.4f}",
        "Q值": "-" if last_q is None else f"{last_q:.4f}",
    })

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
