from __future__ import annotations

import argparse
import os
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


@dataclass
class AgentConfig:
    replay_capacity: int = 100000
    gamma: float = 0.9
    learning_rate: float = 0.00025
    exploration_rate: float = 1.0
    exploration_rate_decay: float = 0.99999975
    exploration_rate_min: float = 0.1


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
    episodes: int = 40000
    batch_size: int = 32
    burnin: int = int(1e5)
    learn_every: int = 3
    sync_every: int = int(1e4)
    record_every: int = 20
    num_envs: int = max(1, (os.cpu_count() or 2) // 2)
    vector: bool = False


@dataclass
class AppConfig:
    env: EnvConfig
    agent: AgentConfig
    artifacts: ArtifactConfig
    training: TrainingConfig


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
    parser.add_argument("--vector", action="store_true", help="启用并行环境训练")
    parser.add_argument("--num-envs", type=int, default=TrainingConfig.num_envs, help="并行环境数量")
    parser.add_argument("--save-root", type=Path, default=ArtifactConfig.save_root, help="checkpoint 根目录")
    parser.add_argument("--checkpoint", type=Path, default=None, help="指定要恢复的 checkpoint")
    parser.add_argument("--no-resume", action="store_true", help="忽略 latest checkpoint，从头开始训练")
    parser.add_argument("--keep-runs", action="store_true", help="额外保留本次训练的 run 目录")
    parser.add_argument("--run-name", default=None, help="指定 run 目录名称")
    parser.add_argument("--record-every", type=int, default=TrainingConfig.record_every, help="每隔多少回合记录指标")
    parser.add_argument("--save-every", type=int, default=ArtifactConfig.save_every, help="每隔多少步保存 checkpoint")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size, help="每次学习采样的 batch 大小")
    parser.add_argument("--burnin", type=int, default=TrainingConfig.burnin, help="开始训练前收集的经验数量")
    parser.add_argument("--replay-capacity", type=int, default=AgentConfig.replay_capacity, help="经验回放容量")
    parser.add_argument("--learning-rate", type=float, default=AgentConfig.learning_rate, help="学习率")
    parser.add_argument("--gamma", type=float, default=AgentConfig.gamma, help="折扣因子")
    parser.add_argument("--exploration-decay", type=float, default=AgentConfig.exploration_rate_decay, help="探索率衰减")
    parser.add_argument("--exploration-min", type=float, default=AgentConfig.exploration_rate_min, help="最小探索率")


def add_play_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.chkpt"), help="要播放的 checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="播放回合数")
    parser.add_argument("--render-mode", default="rgb_array", help="环境渲染模式")
    parser.add_argument("--save-root", type=Path, default=ArtifactConfig.save_root, help="输出根目录")


def config_from_train_args(args) -> AppConfig:
    return AppConfig(
        env=EnvConfig(),
        agent=AgentConfig(
            replay_capacity=args.replay_capacity,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            exploration_rate_decay=args.exploration_decay,
            exploration_rate_min=args.exploration_min,
        ),
        artifacts=ArtifactConfig(
            save_root=args.save_root,
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
            num_envs=max(1, args.num_envs),
            vector=args.vector,
        ),
    )


def config_from_play_args(args) -> AppConfig:
    return AppConfig(
        env=EnvConfig(render_mode=args.render_mode),
        agent=AgentConfig(exploration_rate=0.0, exploration_rate_min=0.0),
        artifacts=ArtifactConfig(save_root=args.save_root, checkpoint=args.checkpoint, resume=True),
        training=TrainingConfig(episodes=args.episodes),
    )
