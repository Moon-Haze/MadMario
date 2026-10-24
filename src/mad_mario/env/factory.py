import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from gymnasium.wrappers import ClipReward, FrameStackObservation as FrameStack, GrayscaleObservation as GrayScaleObservation, RepeatAction, ResizeObservation
from nes_py.wrappers import JoypadSpace

from mad_mario.config import EnvConfig, build_env_name
from mad_mario.env.wrappers import FlagReward, ProgressReward, ScoreReward, StuckHandler


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
    env = RepeatAction(env, num_repeats=config.frame_skip)
    if config.clip_rewards:
        env = ClipReward(env, min_reward=-config.reward_clip_value, max_reward=config.reward_clip_value)
    if config.progress_reward_enabled:
        env = ProgressReward(env, scale=config.progress_reward_scale)
    if config.score_reward_enabled:
        env = ScoreReward(env, scale=config.score_reward_scale)
    if config.flag_reward_enabled:
        env = FlagReward(env, reward_value=config.flag_reward_value)
    if config.stuck_penalty_enabled or config.stuck_rescue_enabled:
        env = StuckHandler(
            env,
            max_stuck_steps=config.stuck_max_steps,
            penalty=config.stuck_penalty,
            progress_epsilon=config.stuck_progress_epsilon,
            penalty_enabled=config.stuck_penalty_enabled,
            rescue_enabled=config.stuck_rescue_enabled,
            rescue_max_steps=config.stuck_rescue_max_steps,
        )
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(config.resize_shape, config.resize_shape))
    env = FrameStack(env, stack_size=config.frame_stack)
    return env
