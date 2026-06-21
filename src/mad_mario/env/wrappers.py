import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from skimage import transform


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
        resize_obs *= 255
        return resize_obs.astype(np.uint8)


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """只返回每第 `skip` 帧。"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """重复执行动作，并累加奖励。"""
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ClipReward(gym.Wrapper):
    def __init__(self, env, clip_value):
        super().__init__(env)
        self.clip_value = float(clip_value)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(np.clip(reward, -self.clip_value, self.clip_value))
        return obs, reward, terminated, truncated, info


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


class NormalizeObservation(gym.ObservationWrapper):
    """将观测从 uint8 [0, 255] 归一化为 float32 [0, 1]。"""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation, dtype=np.float32) / 255.0
