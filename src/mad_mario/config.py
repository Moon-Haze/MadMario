from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnvConfig:
    game: str = "SuperMarioBros"
    world: int = 1
    level: int = 1
    version: int = 0
    levels: list[str] | None = None
    worlds: list[int] | None = None
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


MAX_WORLD = 8


def _parse_world_token(token: str) -> list[int]:
    """解析单个 world token：``3`` → ``[3]``，``1-3`` → ``[1, 2, 3]``。"""
    token = token.strip()
    if "-" in token:
        parts = token.split("-", maxsplit=1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise argparse.ArgumentTypeError(
                f"世界范围格式错误: {token!r}，应使用 1-3 格式"
            )
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"世界范围必须为数字: {token!r}"
            ) from exc
        if start > end:
            raise argparse.ArgumentTypeError(
                f"世界范围起始值 {start} 大于结束值 {end}"
            )
        if not (1 <= start <= MAX_WORLD) or not (1 <= end <= MAX_WORLD):
            raise argparse.ArgumentTypeError(
                f"世界号超出范围，Super Mario 的世界号为 1～{MAX_WORLD}，得到: {token!r}"
            )
        return list(range(start, end + 1))
    else:
        try:
            w = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"世界号格式错误: {token!r}，应为数字或范围如 1-3"
            ) from exc
        if not (1 <= w <= MAX_WORLD):
            raise argparse.ArgumentTypeError(
                f"世界号 {w} 超出范围，Super Mario 的世界号为 1～{MAX_WORLD}"
            )
        return [w]


def _worlds_type(value: str) -> list[int]:
    """argparse type: 解析并校验世界号列表，支持逗号和范围。

    支持格式:
    - ``1,3,5`` — 逗号分隔
    - ``1-4`` — 范围（含两端）
    - ``1-3,5,7-8`` — 混合
    """
    try:
        tokens = [t.strip() for t in value.split(",") if t.strip()]
    except Exception as exc:
        raise argparse.ArgumentTypeError("worlds 参数格式错误") from exc
    if not tokens:
        raise argparse.ArgumentTypeError("worlds 参数不能为空")

    worlds = []
    for token in tokens:
        worlds.extend(_parse_world_token(token))

    if not worlds:
        raise argparse.ArgumentTypeError("worlds 参数不能为空")

    # 去重并保持顺序
    seen = set()
    unique = []
    for w in worlds:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


MAX_LEVEL = 4


def _levels_type(value: str) -> list[str]:
    """argparse type: 解析并校验关卡列表。

    支持两种格式:
    - 纯关卡号如 ``1,2``（与 ``--world`` / ``--worlds`` 组合使用）
    - world-level 格式如 ``1-1,2-1``
    """
    try:
        parts = [v.strip() for v in value.split(",") if v.strip()]
    except Exception as exc:
        raise argparse.ArgumentTypeError("levels 参数格式错误") from exc
    if not parts:
        raise argparse.ArgumentTypeError("levels 参数不能为空")
    for p in parts:
        if "-" in p:
            tokens = p.split("-")
            if len(tokens) != 2:
                raise argparse.ArgumentTypeError(
                    f"关卡格式错误: {p!r}，应使用 world-level 格式如 1-1"
                )
            try:
                w, lv = int(tokens[0]), int(tokens[1])
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"关卡中的世界号和关卡号必须为数字: {p!r}"
                ) from exc
            if not (1 <= w <= MAX_WORLD):
                raise argparse.ArgumentTypeError(
                    f"关卡 {p!r} 中的世界号 {w} 超出范围，Super Mario 的世界号为 1～{MAX_WORLD}"
                )
            if not (1 <= lv <= MAX_LEVEL):
                raise argparse.ArgumentTypeError(
                    f"关卡 {p!r} 中的关卡号 {lv} 超出范围，每世界关卡号为 1～{MAX_LEVEL}"
                )
        else:
            try:
                lv = int(p)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"关卡格式错误: {p!r}，应为数字或 world-level 格式如 1-1"
                ) from exc
            if not (1 <= lv <= MAX_LEVEL):
                raise argparse.ArgumentTypeError(
                    f"关卡号 {lv} 超出范围，每世界关卡号为 1～{MAX_LEVEL}"
                )
    return parts


def build_env_name(game: str, world: int, level: int, version: int = 0) -> str:
    """构建完整的 Gym 环境 ID。"""
    return f"{game}-{world}-{level}-v{version}"


def _expand_level(level_id: str, game: str = "SuperMarioBros", version: int = 0) -> str:
    """将简写关卡名转为完整 env ID，如 1-1 → SuperMarioBros-1-1-v0。"""
    level_id = level_id.strip()
    if "-v" in level_id:
        return level_id
    return build_env_name(game, *map(int, level_id.split("-")), version)


def compatible_save_root(save_root: Path, config: EnvConfig) -> Path:
    if config.worlds and len(config.worlds) > 1:
        worlds_str = "-".join(str(w) for w in config.worlds)
        if config.levels:
            levels_str = _extract_level_numbers(config.levels, config.worlds)
        else:
            levels_str = str(config.level)
        folder = f"worlds-{worlds_str}_levels-{levels_str}-v{config.version}_{config.movement}"
    elif config.levels and len(config.levels) > 1:
        folder = f"multi-v{config.version}_{config.movement}"
    else:
        folder = f"{config.world}-{config.level}-v{config.version}_{config.movement}"
    return save_root / folder


def _extract_level_numbers(levels: list[str], worlds: list[int]) -> str:
    """从完整 env ID 列表中提取关卡号，去重后拼成短字符串。"""
    seen = set()
    numbers = []
    for env_id in levels:
        parts = env_id.split("-")
        if len(parts) >= 2:
            try:
                level_num = int(parts[-2])
            except ValueError:
                continue
            if level_num not in seen:
                seen.add(level_num)
                numbers.append(str(level_num))
    return "-".join(numbers) if numbers else "all"


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
    parser.add_argument("--game", type=str, default=EnvConfig.game, help="游戏名称")
    parser.add_argument("--world", type=int, default=EnvConfig.world, help="世界/大关")
    parser.add_argument("--level", type=int, default=EnvConfig.level, help="关卡")
    parser.add_argument("--version", type=int, default=EnvConfig.version, help="环境版本")
    parser.add_argument(
        "--levels",
        type=_levels_type,
        default=None,
        help="多关卡训练，逗号分隔如 1-1,1-2,1-3（世界号 1～8，关卡号 1～4）",
    )
    parser.add_argument(
        "--worlds",
        type=_worlds_type,
        default=None,
        help="多世界训练，逗号分隔和范围如 1-3,5,7-8（世界号 1～8）；每个世界自动使用关卡 1～4",
    )
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
    parser.add_argument("--game", type=str, default=EnvConfig.game, help="游戏名称")
    parser.add_argument("--world", type=int, default=EnvConfig.world, help="世界/大关")
    parser.add_argument("--level", type=int, default=EnvConfig.level, help="关卡")
    parser.add_argument("--version", type=int, default=EnvConfig.version, help="环境版本")
    parser.add_argument(
        "--movement",
        choices=("right_only", "simple", "complex"),
        default=EnvConfig.movement,
        help="动作空间预设",
    )
    parser.add_argument("--render-mode", default="rgb_array", help="环境渲染模式")
    parser.add_argument("--save-root", type=Path, default=ArtifactConfig.save_root, help="输出根目录")
    parser.add_argument("--flat-save-root", action="store_true", help="直接使用输出根目录，不按兼容参数分子目录")


def _get_level_ids_for_worlds(worlds: list[int]) -> list[str]:
    """根据世界号列表生成所有关卡 1-4 的 ID 列表（笛卡尔积）。"""
    raw_levels = [str(lv) for lv in range(1, MAX_LEVEL + 1)]
    expanded = []
    for w in worlds:
        for rl in raw_levels:
            expanded.append(f"{w}-{rl}")
    return expanded


def config_from_train_args(args) -> AppConfig:
    levels = None
    worlds = None
    if args.worlds:
        worlds = args.worlds
        raw_levels = _get_level_ids_for_worlds(worlds)
        levels = [
            _expand_level(lv, game=args.game, version=args.version)
            for lv in raw_levels
        ]
    elif args.levels:
        levels = [
            _expand_level(level, game=args.game, version=args.version)
            for level in args.levels
        ]
    env_config = EnvConfig(
        game=args.game,
        world=args.world,
        level=args.level,
        version=args.version,
        movement=args.movement,
        levels=levels,
        worlds=worlds,
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
        game=args.game,
        world=args.world,
        level=args.level,
        version=args.version,
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
