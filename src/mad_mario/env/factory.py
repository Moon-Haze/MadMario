from functools import partial

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import FrameStackObservation as FrameStack, GrayscaleObservation as GrayScaleObservation
from nes_py.wrappers import JoypadSpace

from mad_mario.config import EnvConfig
from mad_mario.env.wrappers import NormalizeObservation, ResizeObservation, SkipFrame


def make_mario_env(config: EnvConfig | None = None, render_mode=None):
    config = config or EnvConfig()
    render_mode = config.render_mode if render_mode is None else render_mode

    env = gym_super_mario_bros.make(config.env_name, render_mode=render_mode)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=config.frame_skip)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=config.resize_shape)
    env = NormalizeObservation(env)
    env = FrameStack(env, stack_size=config.frame_stack)
    return env


def make_env_thunk(config: EnvConfig | None = None, render_mode=None):
    return partial(make_mario_env, config=config, render_mode=render_mode)


def make_vector_env(num_envs, config: EnvConfig | None = None):
    return AsyncVectorEnv([make_env_thunk(config=config) for _ in range(num_envs)])
