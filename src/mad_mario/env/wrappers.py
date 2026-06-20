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


class NormalizeObservation(gym.ObservationWrapper):
    """将观测从 uint8 [0, 255] 归一化为 float32 [0, 1]。"""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation, dtype=np.float32) / 255.0
