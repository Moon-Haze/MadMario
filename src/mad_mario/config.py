from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, replace
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
    frame_skip: int = 4
    frame_stack: int = 4
    resize_shape: int = 84
    movement: str = "right_only"
    # 默认关闭：裁剪把 ±1 之外的奖励一律压平，−25 的死亡惩罚被砍成 −1、
    # 每步约 +7 的前进奖励被砍成 +1，原生奖励的区分度基本丢光
    clip_rewards: bool = False
    reward_clip_value: float = 1.0
    stuck_penalty_enabled: bool = True
    stuck_max_steps: int = 120
    stuck_penalty: float = 5.0
    # 窗口内净推进阈值（像素）：窗口内最大 x 相对窗口起点推进不足此值即视为未取得进展
    stuck_progress_epsilon: float = 8.0
    # 卡住后随机接管动作尝试脱困；关闭则回到"达到阈值直接判负"的旧行为
    stuck_rescue_enabled: bool = True
    # 救援预算（步）：用尽仍无进展时，按 stuck_penalty_enabled 决定判负或无缝重开
    stuck_rescue_max_steps: int = 120
    progress_reward_enabled: bool = True
    progress_reward_scale: float = 0.01
    score_reward_enabled: bool = True
    score_reward_scale: float = 0.01
    flag_reward_enabled: bool = True
    flag_reward_value: float = 50.0


def playback_env_config(config: EnvConfig) -> EnvConfig:
    """播放语义的环境配置：观赏，不是继续训练。

    关掉全部奖励塑形（裁剪 / 进度 / 得分 / 通关）与卡住惩罚——视频里要看的是
    模型的原生表现；保留卡住脱困，否则模型卡住时画面会一直僵着。观测变换与
    训练保持一致（形状对不上权重就白加载了）。

    "播放栈是什么"在此单点定义（EnvConfig 的伴生变换；wrapper 组装顺序归
    factory，两层知识各守各的）：命令行播放 config_from_play_args 与训练内
    最优录制 best_playback 都指名调用，不再各自改写六个布尔。
    """
    return replace(
        config,
        clip_rewards=False,
        stuck_penalty_enabled=False,
        stuck_rescue_enabled=True,
        progress_reward_enabled=False,
        score_reward_enabled=False,
        flag_reward_enabled=False,
    )


# ──────────────────────────────────────────────
# SB3 DQN 超参数配置
# ──────────────────────────────────────────────
@dataclass
class SB3Config:
    """映射到 stable-baselines3 DQN 构造参数。"""
    total_timesteps: int = 10_000_000
    learning_rate: float = 0.0000625
    # 注意这是**总转移数**：SB3 会除以并行环境数（buffer_size // n_envs）。
    # 5 万配 24 个环境只剩 2083 条/环境、约 770 个回合，稀疏的通关样本留不住。
    # 10 万约占 5.3 GiB——20 万要 11.3 GiB，实测本机可用内存只有 9.8 GiB，
    # SB3 会直接告警"装不下"。内存宽裕的机器可以往上调。
    buffer_size: int = 100_000
    learning_starts: int = 50_000
    batch_size: int = 64
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4  # 每 N 步训练一次
    gradient_steps: int = 1
    target_update_interval: int = 8_000
    exploration_fraction: float = 0.1
    # 绝对退火步数：全局第几步退到 exploration_final_eps。设了就覆盖
    # exploration_fraction——比例语义会随总步数一起放大（0.1 配 10 亿步 =
    # 1.06 亿步的退火窗口，ε 跑到 8700 万步时还有 0.2），绝对步数不会。
    # None 表示沿用比例。
    exploration_steps: int | None = None
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    max_grad_norm: float = 10.0
    seed: int = 42
    # n 步回报；>1 时 SB3 自动把回放缓冲区换成 NStepReplayBuffer
    n_steps: int = 1
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
    # 每个评估关卡跑几个回合。回合越多，评估均值的抖动越小，"最优"才越可信；
    # 代价是每次评估更慢，而评估是阻塞训练的
    eval_episodes: int = 10
    # 评估覆盖几个训练关卡（从训练关卡集里等距抽取，确定性）。评估总回合数 =
    # 本值 × eval_episodes。置 1 即退回"只评一关"的旧行为
    eval_levels: int = 3
    # 刷新最优模型的门槛：新评估要比已采纳的最优高出这个比例才算数（0.05 = 5%）。
    # 不设门槛的话，评估噪声的每个小波峰都会存一次盘、录一次视频
    best_min_improvement: float = 0.05
    log_interval: int = 20            # 每隔多少 episode 输出日志+刷新图表
    num_envs: int = max(1, (os.cpu_count() or 2) // 2)


# ──────────────────────────────────────────────
# 播放配置
# ──────────────────────────────────────────────
@dataclass
class PlayConfig:
    video_dir: Path | None = None   # None 表示用 save_root/videos
    video_fps: int = 15             # 对应 frame_skip=4 时的真实播放速度
    show: bool = True               # 同时弹窗实时显示；无 GUI 环境会自动降级
    play_on_best: bool = True       # 最优模型刷新到够格时录一次视频（门槛见 best_min_improvement）
    # 视频再套一层时间戳目录，让每场播放各存一份、互不覆盖。训练侧录 best 模型
    # 时置 False：那边的 step_xxx 目录本身已经分好档了
    timestamp_videos: bool = True
    # 按 ε-greedy 采样动作（用 checkpoint 里存下的 exploration_rate）玩几遍，
    # 而不是贪心取最大 Q。0 表示不采样：只贪心地播一遍，这是默认行为——默认该
    # 是"看这个模型最好的表现"。之所以需要它，是因为贪心策略 + 固定初始状态
    # 意味着每次播放都逐帧相同，连跑几遍也看不到新东西；只有引入动作随机性，
    # 轨迹才会真正分岔，而且每遍都值得单独录下来
    sample_runs: int = 0


# ──────────────────────────────────────────────
# 顶层配置
# ──────────────────────────────────────────────
@dataclass
class AppConfig:
    env: EnvConfig
    sb3: SB3Config
    artifacts: SB3ArtifactConfig
    play: PlayConfig = field(default_factory=PlayConfig)


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


# 磁盘发现与目录命名策略已迁往 artifacts.py；这里保留转发导入，组装器与
# 老调用方（trainer / cli / tests）不必一次性改地址
from mad_mario.artifacts import (  # noqa: E402
    _format_worlds,
    compatible_save_root,
    resolve_checkpoint,
)


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
                       help="epsilon 衰减占总步数的比例（设了 --exploration-steps 时被忽略）")
    train.add_argument("--exploration-steps", type=int, default=SB3Config.exploration_steps,
                       help="epsilon 衰减的绝对步数：全局第几步退到 --exploration-final-eps；"
                            "不受总步数放大影响，续训时也按此重新校准")
    train.add_argument("--exploration-initial-eps", type=float, default=SB3Config.exploration_initial_eps,
                       help="初始探索率")
    train.add_argument("--exploration-final-eps", type=float, default=SB3Config.exploration_final_eps,
                       help="最小探索率")
    train.add_argument("--max-grad-norm", type=float, default=SB3Config.max_grad_norm,
                       help="梯度裁剪范数")
    train.add_argument("--seed", type=int, default=SB3Config.seed, help="随机种子")
    train.add_argument("--n-steps", type=int, default=SB3Config.n_steps,
                       help="n 步回报的步数，>1 时启用 NStepReplayBuffer")

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
                     help="每个评估关卡跑几个回合")
    art.add_argument("--eval-levels", type=int, default=SB3ArtifactConfig.eval_levels,
                     help="评估覆盖的训练关卡数（等距抽取）；评估总回合数 = 关卡数 × --eval-episodes")
    art.add_argument("--best-min-improvement", type=float,
                     default=SB3ArtifactConfig.best_min_improvement,
                     help="刷新最优模型的门槛：新评估要比当前最优高出这个比例才算数"
                          "（0.05 = 5%%）；调小则更容易刷新，0 即退回旧行为")
    art.add_argument("--save-freq", type=int, default=SB3ArtifactConfig.save_freq,
                     help="每隔多少步保存一次 checkpoint")
    art.add_argument("--no-play-best", dest="play_on_best", action="store_false",
                     help="最优模型刷新时不录制视频（默认每次刷新都录）")
    art.add_argument("--log-interval", type=int, default=SB3ArtifactConfig.log_interval,
                     help="每隔多少 episode 输出日志")
    art.add_argument("--log-level", choices=("debug", "info", "warning", "error"),
                     default=SB3ArtifactConfig.log_level,
                     help="日志级别（默认 info）")

    # ── 奖励塑形参数 ──
    rw = parser.add_argument_group("奖励塑形")
    # BooleanOptionalAction 同时生成 --reward-clip 与 --no-reward-clip，
    # 旧脚本里的 --no-reward-clip 因此继续可用
    rw.add_argument("--reward-clip", action=argparse.BooleanOptionalAction,
                    default=EnvConfig.clip_rewards,
                    help="奖励裁剪（默认关闭：裁剪把 -25 的死亡惩罚压成 -1，原生奖励的区分度全丢）")
    rw.add_argument("--reward-clip-value", type=float, default=EnvConfig.reward_clip_value,
                    help="奖励裁剪绝对值")
    rw.add_argument("--no-stuck-penalty", action="store_true",
                    help="关闭卡住惩罚与提前结束（不关脱困；要完全关闭卡住处理需同时传 --no-stuck-rescue）")
    rw.add_argument("--stuck-max-steps", type=int, default=EnvConfig.stuck_max_steps,
                    help="连续多少步无净推进后判定卡住")
    rw.add_argument("--stuck-penalty", type=float, default=EnvConfig.stuck_penalty,
                    help="卡住时扣除的奖励")
    rw.add_argument("--no-stuck-rescue", action="store_true",
                    help="关闭卡住后的随机脱困干预（关闭后恢复为达到阈值直接判负）")
    rw.add_argument("--stuck-rescue-max-steps", type=int, default=EnvConfig.stuck_rescue_max_steps,
                    help="卡住后随机脱困的步数预算")
    rw.add_argument("--stuck-progress-epsilon", type=float, default=EnvConfig.stuck_progress_epsilon,
                    help="窗口内净推进阈值（像素），低于此值视为未取得进展")
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
    # nargs="?"：裸写 --sample 等价于 --sample 1，老用法不至于失效；
    # default=None 才能把"没传"和"--sample 0"区分开
    parser.add_argument("--sample", type=int, nargs="?", const=1, default=None,
                        metavar="N",
                        help="按 ε-greedy 采样动作玩 N 遍，每遍各录一个视频（默认 1 遍）")
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
    parser.add_argument("--save-root", type=Path, default=SB3ArtifactConfig.save_root,
                        help="输出根目录")
    parser.add_argument("--flat-save-root", action="store_true",
                        help="直接使用输出根目录，不按兼容参数分子目录")
    parser.add_argument("--video-dir", type=Path, default=None,
                        help="视频输出目录（默认 save_root/videos）")
    parser.add_argument("--video-fps", type=int, default=PlayConfig.video_fps,
                        help="视频帧率，15 对应 frame_skip=4 时的真实播放速度")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="只录制不弹窗；无显示器的环境会自动走这个行为")


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
        clip_rewards=args.reward_clip,
        reward_clip_value=args.reward_clip_value,
        stuck_penalty_enabled=not args.no_stuck_penalty,
        stuck_max_steps=max(1, args.stuck_max_steps),
        stuck_penalty=args.stuck_penalty,
        stuck_progress_epsilon=args.stuck_progress_epsilon,
        stuck_rescue_enabled=not args.no_stuck_rescue,
        stuck_rescue_max_steps=max(1, args.stuck_rescue_max_steps),
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
        # 0 与负数都当作"未启用"，免得算出负比例
        exploration_steps=(max(1, args.exploration_steps)
                           if args.exploration_steps is not None else None),
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        n_steps=max(1, args.n_steps),
    )

    artifact_config = SB3ArtifactConfig(
        save_root=save_root,
        num_envs=max(1, args.num_envs),
        checkpoint=args.checkpoint,
        save_replay_buffer=args.save_replay_buffer,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_levels=max(1, args.eval_levels),
        # 负门槛等于"随便高一点就算最优"，比不设门槛还糟，下界卡在 0
        best_min_improvement=max(0.0, args.best_min_improvement),
        log_interval=args.log_interval,
        log_level=args.log_level,
    )

    return AppConfig(
        env=env_config,
        sb3=sb3_config,
        artifacts=artifact_config,
        play=PlayConfig(play_on_best=args.play_on_best),
    )


def config_from_play_args(args) -> AppConfig:
    levels = None
    worlds = None
    if args.worlds:
        worlds = args.worlds
        raw_levels = _get_level_ids_for_worlds(worlds)
        levels = [_expand_level(lv, game=args.game, version=args.version) for lv in raw_levels]
    elif args.levels:
        levels = [_expand_level(lv, game=args.game, version=args.version) for lv in args.levels]

    # 播放语义（关塑形与惩罚、留脱困）由 playback_env_config 单点定义，这里只指名获取
    env_config = playback_env_config(EnvConfig(
        game=args.game,
        world=args.world,
        level=args.level,
        version=args.version,
        movement=args.movement,
        levels=levels,
        worlds=worlds,
    ))
    save_root = args.save_root if args.flat_save_root else compatible_save_root(args.save_root, env_config)

    checkpoint = args.checkpoint or resolve_checkpoint(
        SB3ArtifactConfig(save_root=save_root, resume=True)
    )

    return AppConfig(
        env=env_config,
        sb3=SB3Config(),
        artifacts=SB3ArtifactConfig(save_root=save_root, checkpoint=checkpoint, resume=True),
        # fps 至少为 1，0 或负数会让编码器直接失败
        play=PlayConfig(
            video_dir=args.video_dir,
            video_fps=max(1, args.video_fps),
            show=args.show,
            # 没传就是 0（贪心一遍）；传了则至少 1 遍，"一遍都不采样"没有意义
            sample_runs=0 if args.sample is None else max(1, args.sample),
        ),
    )
