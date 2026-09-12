from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from mad_mario.config import AppConfig, compatible_save_root, resolve_checkpoint
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
                model = DQN.load(ckpt_path, env=env, device="auto")
                loaded = True
            except Exception as exc:
                logger.warning("checkpoint 加载失败，将从头开始训练。原因: %s", exc)

    if model is None:
        logger.info("从头开始创建 DQN 模型...")
        model = DQN(**dqn_params)

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

    # 评估回调（SB3 内置逻辑：保存 best model + 记录评估日志），并在最优模型
    # 刷新时顺带录一段视频；--no-play-best 可关掉录制，评估本身不受影响
    eval_env = SubprocVecEnv([lambda: Monitor(_env_thunk(config.env)())])
    eval_callback = BestModelPlaybackCallback(
        eval_env,
        best_model_save_path=str(save_root / "best_model"),
        log_path=str(save_root / "eval_logs"),
        eval_freq=max(config.artifacts.eval_freq // config.artifacts.num_envs, 1),
        n_eval_episodes=config.artifacts.eval_episodes,
        deterministic=True,
        # verbose=0 静默评估打印；best model / eval_logs / tensorboard 记录不受影响
        verbose=0,
        train_config=config,
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
        env.close()
