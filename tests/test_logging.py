"""测试 logging/setup.py——装配函数、文件名与日志内容。"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import pytest

from mad_mario.logging.setup import make_log_file_name, setup_logging

LOG_NAME_RE = re.compile(r"train_\d{8}_\d{6}\.log")


@pytest.fixture(autouse=True)
def _isolate_root_logger():
    """隔离全局 root logger，避免测试互相污染。"""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_disabled = root.disabled
    yield
    root.disabled = saved_disabled
    root.setLevel(saved_level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for handler in saved_handlers:
        root.addHandler(handler)


class TestMakeLogFileName:
    def test_name(self):
        now = datetime(2026, 9, 6, 15, 4, 22)
        assert make_log_file_name("train", now) == "train_20260906_150422.log"

    def test_custom_prefix(self):
        now = datetime(2026, 1, 2, 3, 4, 5)
        assert make_log_file_name("eval", now) == "eval_20260102_030405.log"


class TestSetupLogging:
    def test_with_log_dir_writes_file(self, tmp_path: Path):
        log_path = setup_logging(logging.INFO, log_dir=tmp_path)
        assert log_path is not None
        assert LOG_NAME_RE.match(log_path.name)
        assert log_path.parent == tmp_path

        logging.getLogger("mad_mario.trainer").info("中文测试消息")
        assert log_path.exists()
        text = log_path.read_text(encoding="utf-8")
        assert "中文测试消息" in text
        assert "[INFO]" in text
        assert "mad_mario.trainer" in text

    def test_without_log_dir_returns_none(self):
        assert setup_logging(logging.INFO) is None

    def test_without_log_dir_has_terminal_handler(self):
        setup_logging(logging.INFO)
        root = logging.getLogger()
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    def test_level_filter(self, tmp_path: Path, capsys):
        setup_logging(logging.WARNING, log_dir=tmp_path)
        logging.getLogger("mad_mario").info("这条 INFO 不应落盘")
        logging.getLogger("mad_mario").warning("这条 WARNING 应落盘")
        files = list(tmp_path.glob("train_*.log"))
        assert files
        text = files[0].read_text(encoding="utf-8")
        assert "这条 WARNING 应落盘" in text
        assert "这条 INFO 不应落盘" not in text
        err = capsys.readouterr().err
        assert "这条 WARNING 应落盘" in err
        assert "这条 INFO 不应落盘" not in err
