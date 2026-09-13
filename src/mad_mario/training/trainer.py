from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv

from mad_mario.config import (
    AppConfig,
    SB3Config,
    build_env_name,
    compatible_save_root,
    resolve_checkpoint,
)
from mad_mario.env.factory import make_mario_env
from mad_mario.logging.metrics import MarioMetricsCallback
from mad_mario.logging.setup import capture_stdout_to
from mad_mario.training.best_playback import BestModelPlaybackCallback

logger = logging.getLogger(__name__)


def _make_env_thunks(config: AppConfig) -> list[callable]:
    """为并行环境创建 env thunk 列表，支持多关卡分配。"""
    levels = config.env.levels
    num_envs = config.artifacts.num_envs

    if levels and len(levels) > 1:
        # 多关卡训练：每个子环境随机分配一个关卡
        thunks = []
        for _ in range(num_envs):
            level_name = random.choice(levels)
            thunks.append(_env_thunk(config.env, level_name))
        return thunks
    else:
        return [_env_thunk(config.env) for _ in range(num_envs)]


def _env_thunk(env_config, level_name: str | None = None):
    """返回一个无参 callable，创建单个 Mario 环境。"""
    return lambda: make_mario_env(config=env_config, level_name=level_name)


def _select_eval_levels(config: AppConfig) -> list[str]:
    """挑出评估要覆盖的训练关卡，等距抽取。

    评估必须落在训练分布上：``EnvConfig.world`` / ``level`` 默认 1-1，早先
    直接 ``_env_thunk(config.env)`` 建评估环境，等于把 best_model 的选取钉死
    在 1-1 上——多关卡训练时那条曲线先涨后落（实测 233.70 → 38.49），挑出来
    的"最优"只反映单关表现。等距而非随机：评估基线要能跨次比较。
    """
    levels = config.env.levels
    if not levels:
        return [build_env_name(config.env.game, config.env.world,
                               config.env.level, config.env.version)]
    wanted = min(config.artifacts.eval_levels, len(levels))
    if wanted >= len(levels):
        return list(levels)
    stride = len(levels) / wanted
    return [levels[int(i * stride)] for i in range(wanted)]


def _make_eval_thunks(config: AppConfig) -> list[callable]:
    """评估环境：每个被选中的关卡一个 thunk，与训练侧共用 _env_thunk。"""
    return [_env_thunk(config.env, level) for level in _select_eval_levels(config)]


# 续训时以 CLI 为准覆盖存档的超参。SB3 的 DQN.load 优先级是
# 「调用方 kwargs > 存档 data > 类默认值」，而这里过去一个都不传，于是
# --buffer-size / --batch-size / --learning-rate 在续训时全被静默忽略——
# 只有 total_timesteps 生效，因为它是 learn() 的参数而非构造参数。
# 不含 policy / policy_kwargs / env 这类结构性参数：换网络会让存档里的权重
# 对不上号，那该另起一次训练而不是续训。
_RESUMABLE_HYPERPARAMS = (
    "learning_rate", "buffer_size", "learning_starts", "batch_size", "tau",
    "gamma", "train_freq", "gradient_steps", "target_update_interval",
    "exploration_initial_eps", "exploration_final_eps", "max_grad_norm", "n_steps",
)


def _resume_hyperparams(dqn_params: dict[str, Any]) -> dict[str, Any]:
    """从 dqn_params 里挑出续训可以安全覆盖存档的超参。"""
    return {key: dqn_params[key] for key in _RESUMABLE_HYPERPARAMS if key in dqn_params}


def _resolve_exploration_fraction(sb3: SB3Config, num_timesteps: int) -> float:
    """把探索退火窗口折算成 SB3 要的比例。

    没设 ``exploration_steps`` 时直接用 ``exploration_fraction``。设了就换算成
    相对 ``total_timesteps + num_timesteps`` 的比例——分母与 SB3 在
    ``_setup_learn`` 里算的 ``_total_timesteps`` 一致（新训练时 num_timesteps
    为 0，退化成 total_timesteps）。只有用这个分母，``exploration_steps`` 才真
    等价于「全局第几步退到 final_eps」，而不会像比例那样随总步数一起被放大。
    """
    if sb3.exploration_steps is None:
        return sb3.exploration_fraction
    total = sb3.total_timesteps + num_timesteps
    if total <= 0:
        return sb3.exploration_fraction
    return min(1.0, sb3.exploration_steps / total)


def _apply_exploration_schedule(model: DQN, sb3: SB3Config, num_timesteps: int) -> None:
    """按 CLI 意图重建 ε 退火表，并把实际窗口打进日志。

    必须等模型就位后再做：``exploration_steps`` 是绝对步数，要拿到
    ``num_timesteps`` 才能换算。退火表是"看着改了、其实没改"最容易发生的地方，
    所以这里无条件以 CLI 为准，并把结果写进日志，让窗口大小不再是隐形的。
    """
    fraction = _resolve_exploration_fraction(sb3, num_timesteps)
    model.exploration_fraction = fraction
    model.exploration_schedule = LinearSchedule(
        start=sb3.exploration_initial_eps,
        end=sb3.exploration_final_eps,
        end_fraction=fraction,
    )
    total = sb3.total_timesteps + num_timesteps
    window = int(fraction * total)
    # SB3 在 __init__ 里把 exploration_rate 占位成 0.0，真正有意义的 ε 要等
    # _on_step 才算出来（dqn.py:141/184）。日志要报的是此刻真实生效的 ε，
    # 所以按当前进度自己求一次——新训练得 initial_eps，续训得存档进度对应的值。
    model.exploration_rate = model.exploration_schedule(
        1.0 - (num_timesteps / total if total > 0 else 0.0)
    )
    source = "绝对步数" if sb3.exploration_steps is not None else "总步数比例"
    logger.info(
        "ε 退火（%s）：窗口 %s 步（占总量 %.2f%%），当前 ε=%.4f",
        source, f"{window:,}", fraction * 100, model.exploration_rate,
    )


def train(config: AppConfig, log_path: Path | None = None) -> None:
    """使用 Stable-Baselines3 DQN 训练 Mario 智能体。

    - 创建或加载 DQN 模型
    - 每 ``save_freq`` 步自动保存 checkpoint（含经验回放缓冲区）
    - 每 ``log_interval`` 回合输出日志并刷新图表
    - 每 ``eval_freq`` 步执行一次评估
    - 支持 ``Ctrl+C`` 安全中断后恢复

    Args:
        config: 顶层配置。
        log_path: 非空时把训练期间写往 stdout 的文本（SB3 回调输出）
            镜像到该日志文件；默认 ``None``（不镜像）。
    """
    save_root = config.artifacts.save_root
    save_root.mkdir(parents=True, exist_ok=True)

    # ── 1. 创建并行环境 ──
    logger.info("创建 %s 个并行环境...", config.artifacts.num_envs)
    env_fns = _make_env_thunks(config)
    env = SubprocVecEnv(env_fns)

    # ── 2. DQN 模型参数 ──
    policy_kwargs = config.sb3.policy_kwargs or {}
    dqn_params: dict[str, Any] = dict(
        policy="CnnPolicy",
        env=env,
        learning_rate=config.sb3.learning_rate,
        buffer_size=config.sb3.buffer_size,
        learning_starts=config.sb3.learning_starts,
        batch_size=config.sb3.batch_size,
        tau=config.sb3.tau,
        gamma=config.sb3.gamma,
        train_freq=config.sb3.train_freq,
        gradient_steps=config.sb3.gradient_steps,
        target_update_interval=config.sb3.target_update_interval,
        exploration_fraction=config.sb3.exploration_fraction,
        exploration_initial_eps=config.sb3.exploration_initial_eps,
        exploration_final_eps=config.sb3.exploration_final_eps,
        max_grad_norm=config.sb3.max_grad_norm,
        seed=config.sb3.seed,
        n_steps=config.sb3.n_steps,
        device="auto",
        policy_kwargs=policy_kwargs,
        verbose=0,
        stats_window_size=100,
        tensorboard_log=str(save_root / "tensorboard"),
    )

    # ── 3. 创建或加载模型 ──
    model: DQN | None = None
    loaded = False

    if config.artifacts.resume:
        ckpt_path = resolve_checkpoint(config.artifacts)
        if ckpt_path is not None and ckpt_path.exists():
            try:
                logger.info("正在恢复 checkpoint: %s", ckpt_path)
                # 显式传 CLI 超参，否则 SB3 会沿用存档里的值（见 _RESUMABLE_HYPERPARAMS）
                model = DQN.load(
                    ckpt_path, env=env, device="auto", **_resume_hyperparams(dqn_params)
                )
                loaded = True
            except Exception as exc:
                logger.warning("checkpoint 加载失败，将从头开始训练。原因: %s", exc)

    if model is None:
        logger.info("从头开始创建 DQN 模型...")
        model = DQN(**dqn_params)

    # 退火表统一在这里按最终 num_timesteps 校准：新建时它是 0，续训时是存档里
    # 那个数，两条路径共用同一段逻辑
    _apply_exploration_schedule(model, config.sb3, int(model.num_timesteps))
    if loaded:
        logger.info(
            "续训超参以 CLI 为准：lr=%s, buffer=%s, batch=%s, n_steps=%s, γ=%s",
            config.sb3.learning_rate, config.sb3.buffer_size,
            config.sb3.batch_size, config.sb3.n_steps, config.sb3.gamma,
        )

    # ── 4. 回调 ──
    save_freq = max(config.artifacts.save_freq // config.artifacts.num_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(save_root / "snapshots"),
        name_prefix="mario_dqn",
        save_replay_buffer=config.artifacts.save_replay_buffer,
        save_vecnormalize=False,
        verbose=1,
    )

    # 评估回调（SB3 内置的评估与日志逻辑），负责保存 best model，并在最优模型
    # 刷新到够格时把它排进后台录制队列；--no-play-best 可关掉录制，评估本身
    # 不受影响。刷新门槛见 --best-min-improvement
    eval_levels = _select_eval_levels(config)
    eval_env = SubprocVecEnv(
        [lambda thunk=thunk: Monitor(thunk()) for thunk in _make_eval_thunks(config)]
    )
    logger.info(
        "评估覆盖 %d 个关卡（每关 %d 回合）：%s",
        len(eval_levels), config.artifacts.eval_episodes, ", ".join(eval_levels),
    )
    eval_callback = BestModelPlaybackCallback(
        eval_env,
        best_model_save_path=str(save_root / "best_model"),
        log_path=str(save_root / "eval_logs"),
        eval_freq=max(config.artifacts.eval_freq // config.artifacts.num_envs, 1),
        # 按关卡数放大：否则并行评估里回合数会被最快通关的那一关垄断
        n_eval_episodes=max(1, config.artifacts.eval_episodes * len(eval_levels)),
        deterministic=True,
        # verbose=0 静默评估打印；best model / eval_logs / tensorboard 记录不受影响
        verbose=0,
        train_config=config,
        min_improvement=config.artifacts.best_min_improvement,
    )

    # 自定义指标回调（CSV + PNG 图表 + Mario 特有指标）
    metrics_callback = MarioMetricsCallback(
        save_dir=save_root,
        log_interval=config.artifacts.log_interval,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback, metrics_callback])

    # ── 5. 训练（镜像 SB3 文本到 log_path）──
    logger.info(
        "开始训练（总步数=%s, 并行数=%s, 保存间隔=%s 步）",
        config.sb3.total_timesteps,
        config.artifacts.num_envs,
        config.artifacts.save_freq,
    )
    try:
        with capture_stdout_to(log_path):
            try:
                model.learn(
                    total_timesteps=config.sb3.total_timesteps,
                    callback=callbacks,
                    reset_num_timesteps=not loaded,
                    log_interval=1,  # SB3 内部日志频率（每 N 步），和我们的回调独立
                )
            except KeyboardInterrupt:
                logger.info("检测到键盘中断，正在保存最终 checkpoint...")

            # ── 6. 保存 ──
            final_path = str(save_root / "mario_dqn_final.zip")
            model.save(final_path)
            logger.info("最终模型已保存到 %s", final_path)
            logger.info("训练完成。")
    finally:
        # 后台录制线程还在往 mp4 里写：先等它把队列清空再收环境，否则进程带着
        # 写了一半的文件退出，播放器连打开都打不开
        eval_callback.wait_for_recording()
        env.close()
