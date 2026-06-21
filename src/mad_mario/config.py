from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnvConfig:
    env_name: str = "SuperMarioBros-1-1-v0"
    state_dim: tuple[int, int, int] = (4, 84, 84)
    frame_skip: int = 4
    frame_stack: int = 4
    resize_shape: int = 84
    render_mode: str | None = None
    movement: str = "right_only"
    clip_rewards: bool = True
    reward_clip_value: float = 1.0
    stuck_penalty_enabled: bool = True
    stuck_max_steps: int = 120
    stuck_penalty: float = 5.0
    progress_reward_enabled: bool = True
    progress_reward_scale: float = 0.01
    normalize_observation: bool = False


@dataclass
class AgentConfig:
    replay_capacity: int = 100000
    gamma: float = 0.99
    learning_rate: float = 0.0000625
    lr_warmup_steps: int = 10000
    lr_min_ratio: float = 0.1
    exploration_rate: float = 1.0
    exploration_decay_steps: int = int(2e6)
    exploration_rate_min: float = 0.02
    gradient_clip_norm: float = 10.0
    n_step: int = 3
    noisy_std_init: float = 0.5
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = int(2e6)
    per_epsilon: float = 1e-6


@dataclass
class ArtifactConfig:
    save_root: Path = Path("checkpoints")
    checkpoint: Path | None = None
    resume: bool = True
    keep_runs: bool = False
    run_name: str | None = None
    save_every: int = int(5e5)


@dataclass
class TrainingConfig:
    episodes: int = 20000
    batch_size: int = 32
    burnin: int = int(5e4)
    learn_every: int = 4
    sync_every: int = int(8e3)
    record_every: int = 20
    eval_every: int = 50
    eval_episodes: int = 3
    num_envs: int = max(1, (os.cpu_count() or 2) // 2)
    vector: bool = False


@dataclass
class AppConfig:
    env: EnvConfig
    agent: AgentConfig
    artifacts: ArtifactConfig
    training: TrainingConfig


def _slug(value) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return slug or "default"


def compatible_save_root(save_root: Path, config: EnvConfig) -> Path:
    state_dim = "x".join(str(dim) for dim in config.state_dim)
    folder = "_".join(
        (
            f"env-{_slug(config.env_name)}",
            f"movement-{_slug(config.movement)}",
            f"state-{state_dim}",
            f"stack-{config.frame_stack}",
            f"resize-{config.resize_shape}",
        )
    )
    return save_root / folder


def resolve_checkpoint(config: ArtifactConfig) -> Path | None:
    if not config.resume:
        return None
    if config.checkpoint is not None:
        return config.checkpoint

    latest_checkpoint = config.save_root / "latest.chkpt"
    return latest_checkpoint if latest_checkpoint.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练或播放 Super Mario DQN 智能体")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="训练 Mario 智能体")
    add_train_args(train)

    play = subparsers.add_parser("play", help="播放训练好的 Mario 智能体")
    add_play_args(play)
    return parser


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes, help="训练回合数")
    parser.add_argument(
        "--movement",
        choices=("right_only", "simple", "complex"),
        default=EnvConfig.movement,
        help="动作空间预设",
    )
    parser.add_argument("--vector", action="store_true", help="启用并行环境训练")
    parser.add_argument("--num-envs", type=int, default=TrainingConfig.num_envs, help="并行环境数量")
    parser.add_argument("--save-root", type=Path, default=ArtifactConfig.save_root, help="checkpoint 根目录")
    parser.add_argument("--flat-save-root", action="store_true", help="直接使用 checkpoint 根目录，不按兼容参数分子目录")
    parser.add_argument("--checkpoint", type=Path, default=None, help="指定要恢复的 checkpoint")
    parser.add_argument("--no-resume", action="store_true", help="忽略 latest checkpoint，从头开始训练")
    parser.add_argument("--keep-runs", action="store_true", help="额外保留本次训练的 run 目录")
    parser.add_argument("--run-name", default=None, help="指定 run 目录名称")
    parser.add_argument("--record-every", type=int, default=TrainingConfig.record_every, help="每隔多少回合打印指标、刷新图表并保存 checkpoint")
    parser.add_argument("--eval-every", type=int, default=TrainingConfig.eval_every, help="每隔多少回合执行一次贪心评估")
    parser.add_argument("--eval-episodes", type=int, default=TrainingConfig.eval_episodes, help="每次评估运行的回合数")
    parser.add_argument("--save-every", type=int, default=ArtifactConfig.save_every, help="每隔多少步保存 checkpoint")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size, help="每次学习采样的 batch 大小")
    parser.add_argument("--burnin", type=int, default=TrainingConfig.burnin, help="开始训练前收集的经验数量")
    parser.add_argument("--replay-capacity", type=int, default=AgentConfig.replay_capacity, help="经验回放容量")
    parser.add_argument("--learning-rate", type=float, default=AgentConfig.learning_rate, help="学习率")
    parser.add_argument("--lr-warmup-steps", type=int, default=AgentConfig.lr_warmup_steps, help="学习率 warmup 步数")
    parser.add_argument("--lr-min-ratio", type=float, default=AgentConfig.lr_min_ratio, help="最小学习率相对初始学习率的比例")
    parser.add_argument("--gamma", type=float, default=AgentConfig.gamma, help="折扣因子")
    parser.add_argument("--exploration-decay-steps", type=int, default=AgentConfig.exploration_decay_steps, help="epsilon 线性衰减步数")
    parser.add_argument("--exploration-min", type=float, default=AgentConfig.exploration_rate_min, help="最小探索率")
    parser.add_argument("--gradient-clip-norm", type=float, default=AgentConfig.gradient_clip_norm, help="梯度裁剪范数")
    parser.add_argument("--n-step", type=int, default=AgentConfig.n_step, help="N-step return 步数")
    parser.add_argument("--noisy-std-init", type=float, default=AgentConfig.noisy_std_init, help="NoisyNet 噪声标准差初始值（设为 0 使用普通 DuelingDQN + epsilon）")
    parser.add_argument("--no-noisy", action="store_true", help="关闭 NoisyNet，回退到 DuelingDQN + epsilon-greedy")
    parser.add_argument("--per-alpha", type=float, default=AgentConfig.per_alpha, help="PER 优先级指数")
    parser.add_argument("--per-beta-start", type=float, default=AgentConfig.per_beta_start, help="PER beta 初始值")
    parser.add_argument("--per-beta-frames", type=int, default=AgentConfig.per_beta_frames, help="PER beta 退火步数")
    parser.add_argument("--per-epsilon", type=float, default=AgentConfig.per_epsilon, help="PER priority epsilon")
    parser.add_argument("--no-reward-clip", action="store_true", help="关闭奖励裁剪")
    parser.add_argument("--reward-clip-value", type=float, default=EnvConfig.reward_clip_value, help="奖励裁剪绝对值")
    parser.add_argument("--no-stuck-penalty", action="store_true", help="关闭卡住惩罚与提前截断")
    parser.add_argument("--stuck-max-steps", type=int, default=EnvConfig.stuck_max_steps, help="连续多少步无位移后判定卡住")
    parser.add_argument("--stuck-penalty", type=float, default=EnvConfig.stuck_penalty, help="卡住时额外扣除的奖励")
    parser.add_argument("--no-progress-reward", action="store_true", help="关闭向右移动奖励")
    parser.add_argument("--progress-reward-scale", type=float, default=EnvConfig.progress_reward_scale, help="每向右移动一像素的额外奖励系数")
    parser.add_argument("--normalize-observation", action="store_true", help="在环境层归一化观测（默认保持 uint8）")


def add_play_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, default=None, help="要播放的 checkpoint")
    parser.add_argument("--episodes", type=int, default=1, help="播放回合数")
    parser.add_argument(
        "--movement",
        choices=("right_only", "simple", "complex"),
        default=EnvConfig.movement,
        help="动作空间预设",
    )
    parser.add_argument("--render-mode", default="rgb_array", help="环境渲染模式")
    parser.add_argument("--save-root", type=Path, default=ArtifactConfig.save_root, help="输出根目录")
    parser.add_argument("--flat-save-root", action="store_true", help="直接使用输出根目录，不按兼容参数分子目录")


def config_from_train_args(args) -> AppConfig:
    env_config = EnvConfig(
        movement=args.movement,
        clip_rewards=not args.no_reward_clip,
        reward_clip_value=args.reward_clip_value,
        stuck_penalty_enabled=not args.no_stuck_penalty,
        stuck_max_steps=max(1, args.stuck_max_steps),
        stuck_penalty=args.stuck_penalty,
        progress_reward_enabled=not args.no_progress_reward,
        progress_reward_scale=args.progress_reward_scale,
        normalize_observation=args.normalize_observation,
    )
    save_root = args.save_root if args.flat_save_root else compatible_save_root(args.save_root, env_config)

    return AppConfig(
        env=env_config,
        agent=AgentConfig(
            replay_capacity=args.replay_capacity,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            lr_warmup_steps=max(1, args.lr_warmup_steps),
            lr_min_ratio=min(1.0, max(0.0, args.lr_min_ratio)),
            exploration_decay_steps=args.exploration_decay_steps,
            exploration_rate_min=args.exploration_min,
            gradient_clip_norm=args.gradient_clip_norm,
            n_step=max(1, args.n_step),
            noisy_std_init=0.0 if args.no_noisy else max(0.0, args.noisy_std_init),
            per_alpha=args.per_alpha,
            per_beta_start=args.per_beta_start,
            per_beta_frames=max(1, args.per_beta_frames),
            per_epsilon=args.per_epsilon,
        ),
        artifacts=ArtifactConfig(
            save_root=save_root,
            checkpoint=args.checkpoint,
            resume=not args.no_resume,
            keep_runs=args.keep_runs,
            run_name=args.run_name,
            save_every=args.save_every,
        ),
        training=TrainingConfig(
            episodes=args.episodes,
            batch_size=args.batch_size,
            burnin=args.burnin,
            record_every=args.record_every,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            num_envs=max(1, args.num_envs),
            vector=args.vector,
        ),
    )


def config_from_play_args(args) -> AppConfig:
    env_config = EnvConfig(
        render_mode=args.render_mode,
        movement=args.movement,
        clip_rewards=False,
        stuck_penalty_enabled=False,
        progress_reward_enabled=False,
    )
    save_root = args.save_root if args.flat_save_root else compatible_save_root(args.save_root, env_config)

    checkpoint = args.checkpoint or save_root / "latest.chkpt"

    return AppConfig(
        env=env_config,
        agent=AgentConfig(exploration_rate=0.0, exploration_rate_min=0.0),
        artifacts=ArtifactConfig(save_root=save_root, checkpoint=checkpoint, resume=True),
        training=TrainingConfig(episodes=args.episodes),
    )
