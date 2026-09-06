"""标准 logging 装配：终端 + 按时间戳归档的训练文本日志。"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

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


class StdoutTee:
    """把写入同时转发到真实 stdout 与一个镜像日志文件，写入后即时 flush。"""

    def __init__(self, real_stdout, log_path: Path):
        self._real = real_stdout
        self._file = open(log_path, mode="a", encoding="utf-8")
        self._file_ok = True

    def write(self, text: str) -> int:
        self._real.write(text)
        try:
            self._real.flush()
        except OSError:
            pass
        if self._file_ok:
            try:
                self._file.write(text)
                self._file.flush()
            except OSError:
                self._file_ok = False
                self._real.write("\n[logging] 镜像日志文件写入失败，已停止镜像到文件。\n")
        return len(text)

    def flush(self) -> None:
        self._real.flush()
        if self._file_ok:
            self._file.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._file.close()

    def isatty(self) -> bool:
        return self._real.isatty()

    def fileno(self) -> int:
        return self._real.fileno()

    def __getattr__(self, name: str):
        # 其余属性（encoding/buffer 等）委托给真实 stdout
        return getattr(self._real, name)


@contextmanager
def capture_stdout_to(log_path: Path | None) -> Iterator[None]:
    """把写往 stdout 的文本同时镜像到日志文件；``log_path`` 为 ``None`` 时为空操作。"""
    if log_path is None:
        yield
        return
    real_stdout = sys.stdout
    try:
        tee = StdoutTee(real_stdout, log_path)
    except OSError as exc:
        logger.warning("无法打开 stdout 镜像日志文件，本次仅终端输出: %s", exc)
        yield
        return
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = real_stdout
        tee.close()
