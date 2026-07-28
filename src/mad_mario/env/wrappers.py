import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ClipReward, RepeatAction, ResizeObservation


class StuckPenalty(gym.Wrapper):
    def __init__(self, env, max_stuck_steps, penalty, movement_epsilon=0.0):
        super().__init__(env)
        self.max_stuck_steps = int(max_stuck_steps)
        self.penalty = float(penalty)
        self.movement_epsilon = float(movement_epsilon)
        self.last_x_pos = None
        self.last_y_pos = None
        self.max_x_pos = 0.0
        self.stuck_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        self.last_x_pos = self._position_value(info.get("x_pos"))
        self.last_y_pos = self._position_value(info.get("y_pos"))
        self.max_x_pos = self.last_x_pos if self.last_x_pos is not None else 0.0
        self.stuck_steps = 0
        info["max_x_pos"] = self.max_x_pos
        info["stuck_steps"] = self.stuck_steps
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        x_pos = self._position_value(info.get("x_pos"))
        y_pos = self._position_value(info.get("y_pos"))

        if x_pos is not None:
            self.max_x_pos = max(self.max_x_pos, x_pos)

        if x_pos is not None and y_pos is not None:
            if self.last_x_pos is None or self.last_y_pos is None:
                moved = True
            else:
                moved = (
                    abs(x_pos - self.last_x_pos) > self.movement_epsilon
                    or abs(y_pos - self.last_y_pos) > self.movement_epsilon
                )

            if moved:
                self.stuck_steps = 0
                self.last_x_pos = x_pos
                self.last_y_pos = y_pos
            else:
                self.stuck_steps += 1

            if self.stuck_steps >= self.max_stuck_steps and not terminated and not truncated:
                reward = float(reward) - self.penalty
                truncated = True
                info["stuck"] = True

        info["max_x_pos"] = self.max_x_pos
        info["stuck_steps"] = self.stuck_steps
        return obs, reward, terminated, truncated, info

    def _position_value(self, value):
        if value is None:
            return None
        return float(value)


class ProgressReward(gym.Wrapper):
    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = float(scale)
        self.last_x_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_x_pos = info.get("x_pos")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos = info.get("x_pos")
        if x_pos is not None and self.last_x_pos is not None:
            delta = float(x_pos) - self.last_x_pos
            reward = float(reward) + self.scale * max(0.0, delta)
        if x_pos is not None:
            self.last_x_pos = float(x_pos)
        return obs, reward, terminated, truncated, info


class ScoreReward(gym.Wrapper):
    """根据游戏得分（score）变化给予额外奖励。

    追踪 ``info["score"]`` 的变化量，将得分增加转化为额外奖励。
    放在 ``ClipReward`` 之后使用，保证得分变化不被裁剪。
    """

    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = float(scale)
        self.last_score = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_score = info.get("score")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        score = info.get("score")
        if score is not None and self.last_score is not None:
            delta = float(score) - self.last_score
            if delta > 0:
                reward = float(reward) + self.scale * delta
        if score is not None:
            self.last_score = float(score)
        return obs, reward, terminated, truncated, info


class FlagReward(gym.Wrapper):
    """拉旗子通关时给予一次性大额奖励。

    检测 ``info["flag_get"]``，为 True 时添加固定奖励。
    放在所有裁剪类包装器之后，确保奖励值完整保留。
    """

    def __init__(self, env, reward_value=50.0):
        super().__init__(env)
        self.reward_value = float(reward_value)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("flag_get"):
            reward = float(reward) + self.reward_value
        return obs, reward, terminated, truncated, info


