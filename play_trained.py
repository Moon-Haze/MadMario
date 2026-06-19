import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import gym_super_mario_bros
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gymnasium.wrappers import FrameStackObservation as FrameStack
from gymnasium.wrappers import GrayscaleObservation as GrayScaleObservation
from nes_py.wrappers import JoypadSpace

from agent import Mario
from wrappers import NormalizeObservation, ResizeObservation, SkipFrame


# Windows + Python 3.13 下 nes-py 自带的 pyglet human 窗口可能会报错。
# 这里用 rgb_array 取画面，再用 matplotlib 打开窗口显示。
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="rgb_array")

# 动作空间必须和训练时保持一致。
# 注意：这里使用 COMPLEX_MOVEMENT 后，旧的 trained_mario.chkpt（2 动作检查点）不能直接加载。
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# 观测预处理必须和训练时保持一致
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = NormalizeObservation(env)
env = FrameStack(env, stack_size=4)

# 加载训练好的模型
mario = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=Path("checkpoints") / "play",
    checkpoint=Path("trained_mario.chkpt"),
)

# 玩游戏时关闭随机探索，只使用训练好的模型决策
mario.exploration_rate = 0.0
mario.exploration_rate_min = 0.0

episodes = 5

plt.ion()
fig, ax = plt.subplots()
image = None
ax.axis("off")
fig.canvas.manager.set_window_title("训练好的 Mario")

try:
    progress_bar = tqdm(range(1, episodes + 1), desc="播放进度", unit="回合")
    for episode in progress_bar:
        state, _ = env.reset()
        total_reward = 0.0
        step = 0
        flag_get = False

        while True:
            frame = env.render()
            if image is None:
                image = ax.imshow(frame)
            else:
                image.set_data(frame)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

            action = mario.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step += 1

            if done or info.get("flag_get"):
                flag_get = bool(info.get("flag_get", False))
                progress_bar.set_postfix({
                    "步数": step,
                    "奖励": f"{total_reward:.1f}",
                    "通关": flag_get,
                })
                print(
                    f"第 {episode} 回合结束："
                    f"步数={step}，奖励={total_reward:.1f}，是否通关={flag_get}"
                )
                break
finally:
    env.close()
