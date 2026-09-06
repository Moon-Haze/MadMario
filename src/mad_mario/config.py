from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


# ──────────────────────────────────────────────
# 环境配置（不变，保留全部字段）
# ──────────────────────────────────────────────
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
    score_reward_enabled: bool = True
    score_reward_scale: float = 0.01
    flag_reward_enabled: bool = True
    flag_reward_value: float = 50.0


# ──────────────────────────────────────────────
# SB3 DQN 超参数配置
# ──────────────────────────────────────────────
@dataclass
class SB3Config:
    """映射到 stable-baselines3 DQN 构造参数。"""
    total_timesteps: int = 10_000_000
    learning_rate: float = 0.0000625
    buffer_size: int = 50_000
    learning_starts: int = 50_000
    batch_size: int = 32
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4  # 每 N 步训练一次
    gradient_steps: int = 1
    target_update_interval: int = 8_000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    max_grad_norm: float = 10.0
    seed: int = 42
    policy_kwargs: dict | None = None


# ──────────────────────────────────────────────
# 训练工件配置（checkpoint / 日志 / 并行环境）
# ──────────────────────────────────────────────
@dataclass
class SB3ArtifactConfig:
    save_root: Path = Path("checkpoints")
    checkpoint: Path | None = None
    resume: bool = True
    run_name: str | None = None
    log_level: str = "info"            # logging 级别：debug/info/warning/error
    save_freq: int = 100_000          # 每隔多少步保存一次 checkpoint
    save_replay_buffer: bool = False  # 保存经验回放缓冲区（文件很大）
    eval_freq: int = 10_000           # 每隔多少步执行一次评估
    eval_episodes: int = 3
    log_interval: int = 20            # 每隔多少 episode 输出日志+刷新图表
    num_envs: int = max(1, (os.cpu_count() or 2) // 2)


# ──────────────────────────────────────────────
# 顶层配置
# ──────────────────────────────────────────────
@dataclass
class AppConfig:
    env: EnvConfig
    sb3: SB3Config
    artifacts: SB3ArtifactConfig


# ──────────────────────────────────────────────
# 世界 / 关卡解析工具（不变）
# ──────────────────────────────────────────────
MAX_WORLD = 8
MAX_LEVEL = 4


def _parse_world_token(token: str) -> list[int]:
    token = token.strip()
    if "-" in token:
        parts = token.split("-", maxsplit=1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise argparse.ArgumentTypeError(f"世界范围格式错误: {token!r}，应使用 1-3 格式")
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"世界范围必须为数字: {token!r}") from exc
        if start > end:
            raise argparse.ArgumentTypeError(f"世界范围起始值 {start} 大于结束值 {end}")
        if not (1 <= start <= MAX_WORLD) or not (1 <= end <= MAX_WORLD):
            raise argparse.ArgumentTypeError(
                f"世界号超出范围，Super Mario 的世界号为 1～{MAX_WORLD}，得到: {token!r}"
            )
        return list(range(start, end + 1))
    else:
        try:
            w = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"世界号格式错误: {token!r}，应为数字或范围如 1-3") from exc
        if not (1 <= w <= MAX_WORLD):
            raise argparse.ArgumentTypeError(f"世界号 {w} 超出范围，Super Mario 的世界号为 1～{MAX_WORLD}")
        return [w]


def _worlds_type(value: str) -> list[int]:
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
    seen = set()
    unique = []
    for w in worlds:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


def _levels_type(value: str) -> list[str]:
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
                raise argparse.ArgumentTypeError(f"关卡格式错误: {p!r}，应使用 world-level 格式如 1-1")
            try:
                w, lv = int(tokens[0]), int(tokens[1])
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"关卡中的世界号和关卡号必须为数字: {p!r}") from exc
            if not (1 <= w <= MAX_WORLD):
                raise argparse.ArgumentTypeError(f"关卡 {p!r} 中的世界号 {w} 超出范围，世界号为 1～{MAX_WORLD}")
            if not (1 <= lv <= MAX_LEVEL):
                raise argparse.ArgumentTypeError(f"关卡 {p!r} 中的关卡号 {lv} 超出范围，关卡号为 1～{MAX_LEVEL}")
        else:
            try:
                lv = int(p)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"关卡格式错误: {p!r}，应为数字或 world-level 格式如 1-1") from exc
            if not (1 <= lv <= MAX_LEVEL):
                raise argparse.ArgumentTypeError(f"关卡号 {lv} 超出范围，关卡号为 1～{MAX_LEVEL}")
    return parts


def build_env_name(game: str, world: int, level: int, version: int = 0) -> str:
    return f"{game}-{world}-{level}-v{version}"


def _expand_level(level_id: str, game: str = "SuperMarioBros", version: int = 0) -> str:
    level_id = level_id.strip()
    if "-v" in level_id:
        return level_id
    return build_env_name(game, *map(int, level_id.split("-")), version)


def _format_worlds(worlds: list[int]) -> str:
    """将世界号列表格式化为紧凑字符串。

    连续范围用 ``-``（如 1-8），非连续用 ``,``（如 1,3,5）。
    单个值直接返回数字。
    """
    if len(worlds) == 1:
        return str(worlds[0])
    # 检查是否连续
    is_consecutive = all(worlds[i] + 1 == worlds[i + 1] for i in range(len(worlds) - 1))
    if is_consecutive:
        return f"{worlds[0]}-{worlds[-1]}"
    return ",".join(str(w) for w in worlds)


def compatible_save_root(save_root: Path, config: EnvConfig) -> Path:
    if config.worlds and len(config.worlds) > 1:
        worlds_str = _format_worlds(config.worlds)
        folder = f"worlds-{worlds_str}-v{config.version}_{config.movement}"
    elif config.levels and len(config.levels) > 1:
        folder = f"multi-v{config.version}_{config.movement}"
    else:
        folder = f"{config.world}-{config.level}-v{config.version}_{config.movement}"
    return save_root / folder


def resolve_checkpoint(config: SB3ArtifactConfig) -> Path | None:
    """解析 SB3 checkpoint 路径。

    SB3 使用 .zip 格式，查找顺序：
    1. config.checkpoint（用户指定）
    2. save_root / "mario_dqn_final.zip"
    3. save_root 中步数最大的 mario_dqn_*_steps.zip
    """
    if not config.resume:
        return None
    if config.checkpoint is not None:
        return config.checkpoint

    save_root = config.save_root
    # 尝试最终 checkpoint
    final = save_root / "mario_dqn_final.zip"
    if final.exists():
        return final

    # 尝试带步数的 checkpoint（取最新）
    step_zips = list(save_root.glob("mario_dqn_*_steps.zip"))
    if step_zips:
        # 按修改时间取最新的
        step_zips.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return step_zips[0]

    return None


# ──────────────────────────────────────────────
# 配置文件加载
# ──────────────────────────────────────────────


def _flatten_config(raw: dict) -> dict:
    """将嵌套分组配置展平为扁平字典。

    支持两种格式：
        {"world": 1, "learning_rate": 0.001}          # 扁平
        {"env": {"world": 1}, "train": {"lr": ...}}  # 分组
    """
    flat = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def load_config_file(path: str) -> dict:
    """加载 JSON 配置文件并返回扁平化配置字典。

    Args:
        path: 配置文件路径（.json）。

    Returns:
        展平后的配置字典，key 对应 argparse dest 名称。

    Raises:
        SystemExit: 文件不存在或 JSON 解析失败时退出。
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"错误：配置文件不存在: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"错误：配置文件 JSON 解析失败: {e}")

    if not isinstance(raw, dict):
        raise SystemExit("错误：配置文件必须是一个 JSON 对象")

    return _flatten_config(raw)


def _extract_config_path() -> str | None:
    """从 ``sys.argv`` 中提取 ``--config`` 路径，不触发完整解析。"""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _apply_config_defaults(parser: argparse.ArgumentParser, config: dict) -> None:
    """将配置项作为默认值应用到所有子解析器。

    这样 argparse 解析时，CLI 参数自然覆盖 config 中的对应值。
    """
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):  # type: ignore[attr-defined]
            for subparser in action.choices.values():
                subparser.set_defaults(**config)


# ──────────────────────────────────────────────
# CLI 解析器
# ──────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练或播放 Super Mario DQN 智能体（基于 Stable-Baselines3）")
    parser.add_argument(
        "--config", type=str, default=None,
        help="JSON 配置文件路径。命令行中显式设置的参数会覆盖配置文件中的对应值。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_train_args(subparsers.add_parser("train", help="训练 Mario 智能体"))
    add_play_args(subparsers.add_parser("play", help="播放训练好的 Mario 智能体"))
    return parser


def add_train_args(parser: argparse.ArgumentParser) -> None:
    # ── 环境参数 ──
    env = parser.add_argument_group("环境设置")
    env.add_argument("--game", type=str, default=EnvConfig.game, help="游戏名称")
    env.add_argument("--world", type=int, default=EnvConfig.world, help="世界/大关")
    env.add_argument("--level", type=int, default=EnvConfig.level, help="关卡")
    env.add_argument("--version", type=int, default=EnvConfig.version, help="环境版本")
    env.add_argument("--levels", type=_levels_type, default=None,
                     help="多关卡，逗号分隔如 1-1,1-2")
    env.add_argument("--worlds", type=_worlds_type, default=None,
                     help="多世界，逗号分隔和范围如 1-3,5,7-8")
    env.add_argument("--movement", choices=("right_only", "simple", "complex"),
                     default=EnvConfig.movement, help="动作空间预设")

    # ── 训练参数 ──
    train = parser.add_argument_group("训练设置")
    train.add_argument("--timesteps", "--total-timesteps", type=int, default=SB3Config.total_timesteps,
                       dest="total_timesteps", help="总训练步数")
    train.add_argument("--learning-rate", type=float, default=SB3Config.learning_rate,
                       help="学习率")
    train.add_argument("--batch-size", type=int, default=SB3Config.batch_size,
                       help="batch 大小")
    train.add_argument("--gamma", type=float, default=SB3Config.gamma,
                       help="折扣因子")
    train.add_argument("--buffer-size", type=int, default=SB3Config.buffer_size,
                       help="经验回放容量")
    train.add_argument("--learning-starts", type=int, default=SB3Config.learning_starts,
                       help="开始学习前收集的步数")
    train.add_argument("--train-freq", type=int, default=SB3Config.train_freq,
                       help="每隔多少步训练一次")
    train.add_argument("--target-update-interval", type=int, default=SB3Config.target_update_interval,
                       help="目标网络更新间隔")
    train.add_argument("--exploration-fraction", type=float, default=SB3Config.exploration_fraction,
                       help="epsilon 衰减占总步数的比例")
    train.add_argument("--exploration-initial-eps", type=float, default=SB3Config.exploration_initial_eps,
                       help="初始探索率")
    train.add_argument("--exploration-final-eps", type=float, default=SB3Config.exploration_final_eps,
                       help="最小探索率")
    train.add_argument("--max-grad-norm", type=float, default=SB3Config.max_grad_norm,
                       help="梯度裁剪范数")
    train.add_argument("--seed", type=int, default=SB3Config.seed, help="随机种子")

    # ── 工件参数 ──
    art = parser.add_argument_group("保存与恢复")
    art.add_argument("--save-root", type=Path, default=SB3ArtifactConfig.save_root,
                     help="checkpoint 根目录")
    art.add_argument("--flat-save-root", action="store_true",
                     help="直接使用 checkpoint 根目录，不按兼容参数分子目录")
    art.add_argument("--run-name", default=None, help="本次训练的名称")
    art.add_argument("--checkpoint", type=Path, default=None, help="指定要恢复的 checkpoint")
    art.add_argument("--save-replay-buffer", action="store_true",
                     help="保存经验回放缓冲区（注意：文件可能很大）")
    art.add_argument("--num-envs", type=int, default=SB3ArtifactConfig.num_envs,
                     help="并行环境数量（默认 CPU 核数/2）")
    art.add_argument("--eval-freq", type=int, default=SB3ArtifactConfig.eval_freq,
                     help="每隔多少步执行评估")
    art.add_argument("--eval-episodes", type=int, default=SB3ArtifactConfig.eval_episodes,
                     help="每次评估的回合数")
    art.add_argument("--log-interval", type=int, default=SB3ArtifactConfig.log_interval,
                     help="每隔多少 episode 输出日志")
    art.add_argument("--log-level", choices=("debug", "info", "warning", "error"),
                     default=SB3ArtifactConfig.log_level,
                     help="日志级别（默认 info）")

    # ── 奖励塑形参数 ──
    rw = parser.add_argument_group("奖励塑形")
    rw.add_argument("--no-reward-clip", action="store_true", help="关闭奖励裁剪")
    rw.add_argument("--reward-clip-value", type=float, default=EnvConfig.reward_clip_value,
                    help="奖励裁剪绝对值")
    rw.add_argument("--no-stuck-penalty", action="store_true", help="关闭卡住惩罚")
    rw.add_argument("--stuck-max-steps", type=int, default=EnvConfig.stuck_max_steps,
                    help="连续多少步无位移后判定卡住")
    rw.add_argument("--stuck-penalty", type=float, default=EnvConfig.stuck_penalty,
                    help="卡住时扣除的奖励")
    rw.add_argument("--no-progress-reward", action="store_true", help="关闭向右移动奖励")
    rw.add_argument("--progress-reward-scale", type=float, default=EnvConfig.progress_reward_scale,
                    help="每向右移动一像素的奖励系数")
    rw.add_argument("--no-score-reward", action="store_true", help="关闭得分变化奖励")
    rw.add_argument("--score-reward-scale", type=float, default=EnvConfig.score_reward_scale,
                    help="每增加 1 游戏得分的奖励系数")
    rw.add_argument("--no-flag-reward", action="store_true", help="关闭旗杆通关奖励")
    rw.add_argument("--flag-reward-value", type=float, default=EnvConfig.flag_reward_value,
                    help="旗杆通关奖励值")


def add_play_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, default=None, help="要播放的 checkpoint")
    parser.add_argument("--episodes", type=int, default=1, help="播放回合数")
    parser.add_argument("--game", type=str, default=EnvConfig.game, help="游戏名称")
    parser.add_argument("--world", type=int, default=EnvConfig.world, help="世界/大关")
    parser.add_argument("--level", type=int, default=EnvConfig.level, help="关卡")
    parser.add_argument("--version", type=int, default=EnvConfig.version, help="环境版本")
    parser.add_argument("--levels", type=_levels_type, default=None,
                        help="多关卡播放，逗号分隔如 1-1,1-2,1-3")
    parser.add_argument("--worlds", type=_worlds_type, default=None,
                        help="多世界播放，逗号分隔和范围如 1-3,5,7-8")
    parser.add_argument("--movement", choices=("right_only", "simple", "complex"),
                        default=EnvConfig.movement, help="动作空间预设")
    parser.add_argument("--render-mode", default="rgb_array", help="环境渲染模式")
    parser.add_argument("--save-root", type=Path, default=SB3ArtifactConfig.save_root,
                        help="输出根目录")
    parser.add_argument("--flat-save-root", action="store_true",
                        help="直接使用输出根目录，不按兼容参数分子目录")


# ──────────────────────────────────────────────
# 配置构建函数
# ──────────────────────────────────────────────
def _get_level_ids_for_worlds(worlds: list[int]) -> list[str]:
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
        levels = [_expand_level(lv, game=args.game, version=args.version) for lv in raw_levels]
    elif args.levels:
        levels = [_expand_level(lv, game=args.game, version=args.version) for lv in args.levels]

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
        score_reward_enabled=not args.no_score_reward,
        score_reward_scale=args.score_reward_scale,
        flag_reward_enabled=not args.no_flag_reward,
        flag_reward_value=args.flag_reward_value,
    )
    save_root = args.save_root if args.flat_save_root else compatible_save_root(args.save_root, env_config)

    sb3_config = SB3Config(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )

    artifact_config = SB3ArtifactConfig(
        save_root=save_root,
        num_envs=max(1, args.num_envs),
        checkpoint=args.checkpoint,
        save_replay_buffer=args.save_replay_buffer,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        log_interval=args.log_interval,
        log_level=args.log_level,
    )

    return AppConfig(env=env_config, sb3=sb3_config, artifacts=artifact_config)


def config_from_play_args(args) -> AppConfig:
    levels = None
    worlds = None
    if args.worlds:
        worlds = args.worlds
        raw_levels = _get_level_ids_for_worlds(worlds)
        levels = [_expand_level(lv, game=args.game, version=args.version) for lv in raw_levels]
    elif args.levels:
        levels = [_expand_level(lv, game=args.game, version=args.version) for lv in args.levels]

    env_config = EnvConfig(
        game=args.game,
        world=args.world,
        level=args.level,
        version=args.version,
        render_mode=args.render_mode,
        movement=args.movement,
        levels=levels,
        worlds=worlds,
        clip_rewards=False,
        stuck_penalty_enabled=False,
        progress_reward_enabled=False,
        score_reward_enabled=False,
        flag_reward_enabled=False,
    )
    save_root = args.save_root if args.flat_save_root else compatible_save_root(args.save_root, env_config)

    checkpoint = args.checkpoint or resolve_checkpoint(
        SB3ArtifactConfig(save_root=save_root, resume=True)
    )

    return AppConfig(
        env=env_config,
        sb3=SB3Config(),
        artifacts=SB3ArtifactConfig(save_root=save_root, checkpoint=checkpoint, resume=True),
    )
