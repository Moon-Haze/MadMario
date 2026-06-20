from functools import partial

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import FrameStackObservation as FrameStack, GrayscaleObservation as GrayScaleObservation
from nes_py.wrappers import JoypadSpace

from mad_mario.config import EnvConfig
from mad_mario.env.wrappers import ClipReward, NormalizeObservation, ResizeObservation, SkipFrame


MOVEMENT_ACTIONS = {
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}


def make_mario_env(config: EnvConfig | None = None, render_mode=None):
    config = config or EnvConfig()
    render_mode = config.render_mode if render_mode is None else render_mode

    env = gym_super_mario_bros.make(config.env_name, render_mode=render_mode)
    try:
        movement_actions = MOVEMENT_ACTIONS[config.movement]
    except KeyError as exc:
        choices = ", ".join(MOVEMENT_ACTIONS)
        raise ValueError(f"未知动作空间预设: {config.movement}，可选值: {choices}") from exc
    env = JoypadSpace(env, movement_actions)
    env = SkipFrame(env, skip=config.frame_skip)
    if config.clip_rewards:
        env = ClipReward(env, clip_value=config.reward_clip_value)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=config.resize_shape)
    if config.normalize_observation:
        env = NormalizeObservation(env)
    env = FrameStack(env, stack_size=config.frame_stack)
    return env


def make_env_thunk(config: EnvConfig | None = None, render_mode=None):
    return partial(make_mario_env, config=config, render_mode=render_mode)


def make_vector_env(num_envs, config: EnvConfig | None = None):
    return AsyncVectorEnv([make_env_thunk(config=config) for _ in range(num_envs)])
