"""训练工件的磁盘发现与目录命名策略。

config.py 管配置的形状，这里管"去磁盘上找什么、目录叫什么"——两类知识
分家后，配置模块不再夹带 I/O 策略。

只依赖 config 的类型标注（TYPE_CHECKING 导入），运行时零导入：config 的
组装器要调用这里的函数，反向依赖会在 import 时成环。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mad_mario.config import EnvConfig, SB3ArtifactConfig


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


def _checkpoint_steps(path: Path) -> int:
    """从 ``mario_dqn_<steps>_steps.zip`` 文件名解析训练步数；解析失败返回 -1。"""
    stem = path.stem
    if not (stem.startswith("mario_dqn_") and stem.endswith("_steps")):
        return -1
    try:
        return int(stem.split("_")[2])
    except (IndexError, ValueError):
        return -1


def resolve_checkpoint(config: SB3ArtifactConfig) -> Path | None:
    """解析 SB3 checkpoint 路径。

    SB3 使用 .zip 格式，查找顺序：
    1. config.checkpoint（用户指定）
    2. save_root 或 save_root/snapshots/ 中步数最大的 mario_dqn_*_steps.zip
    3. save_root / "mario_dqn_final.zip"（仅在没有周期档时兜底）
    """
    if not config.resume:
        return None
    if config.checkpoint is not None:
        return config.checkpoint

    save_root = config.save_root

    # 周期档可能位于 save_root 或 save_root/snapshots/ 子目录，选训练步数最大者
    # （步数比修改时间更能代表训练进度，也避免同目录混入多条训练序列时挑错档）
    candidates = list(save_root.glob("mario_dqn_*_steps.zip"))
    snapshots_dir = save_root / "snapshots"
    if snapshots_dir.is_dir():
        candidates.extend(snapshots_dir.glob("mario_dqn_*_steps.zip"))
    if candidates:
        return max(candidates, key=_checkpoint_steps)

    # 没有周期档时，回退到收尾生成的最终 checkpoint
    final = save_root / "mario_dqn_final.zip"
    if final.exists():
        return final

    return None
