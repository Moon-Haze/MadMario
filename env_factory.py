from functools import partial

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import FrameStackObservation as FrameStack, GrayscaleObservation as GrayScaleObservation
from nes_py.wrappers import JoypadSpace

from wrappers import NormalizeObservation, ResizeObservation, SkipFrame


ENV_NAME = "SuperMarioBros-1-1-v0"


def make_mario_env(render_mode=None):
    env = gym_super_mario_bros.make(ENV_NAME, render_mode=render_mode)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = NormalizeObservation(env)
    env = FrameStack(env, stack_size=4)
    return env


def make_env_thunk(render_mode=None):
    return partial(make_mario_env, render_mode=render_mode)


def make_vector_env(num_envs):
    return AsyncVectorEnv([make_env_thunk() for _ in range(num_envs)])
