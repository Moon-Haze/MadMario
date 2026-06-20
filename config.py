from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class TrainingConfig:
    episodes: int = 40000
    state_dim: tuple[int, int, int] = (4, 84, 84)
    replay_capacity: int = 100000
    batch_size: int = 32
    burnin: int = int(1e5)
    learn_every: int = 3
    sync_every: int = int(1e4)
    save_every: int = int(5e5)
    record_every: int = 20
    num_envs: int = max(1, (os.cpu_count() or 2) // 2)
    save_root: Path = Path("checkpoints")
    checkpoint: Path | None = None
    resume: bool = True

    def with_updates(self, **kwargs):
        return replace(self, **kwargs)


def resolve_checkpoint(config: TrainingConfig) -> Path | None:
    if not config.resume:
        return None
    if config.checkpoint is not None:
        return config.checkpoint

    latest_checkpoint = config.save_root / "latest.chkpt"
    return latest_checkpoint if latest_checkpoint.exists() else None


def build_arg_parser(vector: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 Super Mario DQN 智能体")
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes, help="训练回合数")
    parser.add_argument("--save-root", type=Path, default=TrainingConfig.save_root, help="checkpoint 根目录")
    parser.add_argument("--checkpoint", type=Path, default=None, help="指定要恢复的 checkpoint")
    parser.add_argument("--no-resume", action="store_true", help="忽略 latest checkpoint，从头开始训练")
    parser.add_argument("--replay-capacity", type=int, default=TrainingConfig.replay_capacity, help="经验回放容量")
    parser.add_argument("--burnin", type=int, default=TrainingConfig.burnin, help="开始训练前收集的经验数量")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size, help="每次学习采样的 batch 大小")
    if vector:
        parser.add_argument("--num-envs", type=int, default=TrainingConfig.num_envs, help="并行环境数量")
    return parser


def config_from_args(args, vector: bool = False) -> TrainingConfig:
    updates = dict(
        episodes=args.episodes,
        save_root=args.save_root,
        checkpoint=args.checkpoint,
        resume=not args.no_resume,
        replay_capacity=args.replay_capacity,
        burnin=args.burnin,
        batch_size=args.batch_size,
    )
    if vector:
        updates["num_envs"] = max(1, args.num_envs)
    return TrainingConfig(**updates)
