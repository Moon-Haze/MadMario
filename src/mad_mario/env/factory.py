from functools import partial

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import FrameStackObservation as FrameStack, GrayscaleObservation as GrayScaleObservation
from nes_py.wrappers import JoypadSpace

from mad_mario.config import EnvConfig, build_env_name
from mad_mario.env.wrappers import ClipReward, NormalizeObservation, ProgressReward, ResizeObservation, SkipFrame, StuckPenalty


MOVEMENT_ACTIONS = {
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}


def make_mario_env(config: EnvConfig | None = None, render_mode=None, level_name: str | None = None):
    config = config or EnvConfig()
    render_mode = config.render_mode if render_mode is None else render_mode

    if level_name:
        env_name = level_name
    else:
        env_name = build_env_name(config.game, config.world, config.level, config.version)
    env = gym_super_mario_bros.make(env_name, render_mode=render_mode)
    try:
        movement_actions = MOVEMENT_ACTIONS[config.movement]
    except KeyError as exc:
        choices = ", ".join(MOVEMENT_ACTIONS)
        raise ValueError(f"未知动作空间预设: {config.movement}，可选值: {choices}") from exc
    env = JoypadSpace(env, movement_actions)
    env = SkipFrame(env, skip=config.frame_skip)
    if config.clip_rewards:
        env = ClipReward(env, clip_value=config.reward_clip_value)
    if config.progress_reward_enabled:
        env = ProgressReward(env, scale=config.progress_reward_scale)
    if config.stuck_penalty_enabled:
        env = StuckPenalty(
            env,
            max_stuck_steps=config.stuck_max_steps,
            penalty=config.stuck_penalty,
        )
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=config.resize_shape)
    if config.normalize_observation:
        env = NormalizeObservation(env)
    env = FrameStack(env, stack_size=config.frame_stack)
    return env


def make_env_thunk(config: EnvConfig | None = None, render_mode=None, level_name: str | None = None):
    return partial(make_mario_env, config=config, render_mode=render_mode, level_name=level_name)


def make_vector_env(num_envs, config: EnvConfig | None = None):
    levels = _select_levels(num_envs, config)
    return AsyncVectorEnv([make_env_thunk(config=config, level_name=level) for level in levels])


def _select_levels(num_envs, config):
    """为并行环境的每个子环境分配关卡。"""
    if config and config.levels and len(config.levels) > 1:
        import random
        return [random.choice(config.levels) for _ in range(num_envs)]
    return [None] * num_envs
