"""标准 logging 装配：终端 + 按时间戳归档的训练文本日志。"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)


def make_log_file_name(prefix: str, now: datetime | None = None) -> str:
    """生成 ``<prefix>_YYYYMMDD_HHMMSS.log`` 风格的文件名。"""
    now = now or datetime.now()
    return f"{prefix}_{now:%Y%m%d_%H%M%S}.log"


def setup_logging(
    level: int = logging.INFO,
    log_dir: Path | None = None,
    prefix: str = "train",
    fmt: str = DEFAULT_FORMAT,
) -> Path | None:
    """装配 root logger：终端(stderr) + 可选按时间戳命名的文本文件。可重入。

    Args:
        level: 全局日志级别。
        log_dir: 非空时在此目录创建 ``<prefix>_YYYYMMDD_HHMMSS.log``。
        prefix: 日志文件名的前缀。
        fmt: 日志格式。

    Returns:
        日志文件路径；``log_dir`` 为 ``None`` 或文件创建失败时返回 ``None``。
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(level)
    formatter = logging.Formatter(fmt, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_dir is None:
        return None

    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / make_log_file_name(prefix)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        return log_path
    except OSError as exc:
        root.warning("无法创建日志文件，本次仅输出到终端: %s", exc)
        return None
