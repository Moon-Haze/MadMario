"""测试 logging/setup.py——装配函数、文件名与日志内容。"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

from mad_mario.logging.setup import StdoutTee, capture_stdout_to, make_log_file_name, setup_logging

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


class TestCaptureStdoutTo:
    def test_mirrors_to_file_and_real_stdout(self, tmp_path: Path, capsys):
        log_path = tmp_path / "out.log"
        with capture_stdout_to(log_path):
            print("来自 stdout 的一行")
        out = capsys.readouterr().out
        assert "来自 stdout 的一行" in out
        assert log_path.read_text(encoding="utf-8") == "来自 stdout 的一行\n"

    def test_none_is_noop(self, tmp_path: Path):
        original = sys.stdout
        with capture_stdout_to(None):
            assert sys.stdout is original
            print("普通输出")
        assert sys.stdout is original
        assert list(tmp_path.glob("*.log")) == []

    def test_restores_stdout(self, tmp_path: Path):
        original = sys.stdout
        with capture_stdout_to(tmp_path / "out.log"):
            assert sys.stdout is not original
        assert sys.stdout is original


class _FlakyFlushStdout:
    """write 正常、flush 抛 OSError 的 fake stdout（模拟终端管道断开）。"""

    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, text: str) -> int:
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        raise OSError("broken pipe")


class TestStdoutTeeFlushGuard:
    def test_flush_swallows_real_stdout_oserror(self, tmp_path: Path):
        real = _FlakyFlushStdout()
        log_path = tmp_path / "out.log"
        tee = StdoutTee(real, log_path)
        tee.write("镜像内容\n")  # write 内部已保护 real.flush
        tee.flush()  # 真实 stdout flush 抛 OSError 不应向外传播
        tee.close()  # close 经 flush() 链，同样不应抛异常
        assert log_path.read_text(encoding="utf-8") == "镜像内容\n"

    def test_capture_stdout_survives_real_flush_oserror(self, tmp_path: Path, monkeypatch):
        real = _FlakyFlushStdout()
        monkeypatch.setattr(sys, "stdout", real)
        log_path = tmp_path / "out.log"
        with capture_stdout_to(log_path):
            print("来自 stdout 的一行")
        # 退出时不抛异常，sys.stdout 还原，镜像文件内容仍正确写入
        assert sys.stdout is real
        assert log_path.read_text(encoding="utf-8") == "来自 stdout 的一行\n"
        assert "".join(real.writes) == "来自 stdout 的一行\n"


class TestTrainerMigration:
    def test_trainer_off_tqdm_and_wired_for_capture(self):
        repo = Path(__file__).resolve().parents[1]
        src = (repo / "src" / "mad_mario" / "training" / "trainer.py").read_text(encoding="utf-8")
        assert "tqdm" not in src
        assert "capture_stdout_to" in src
        assert "log_path: Path | None = None" in src

    def test_completion_log_after_save_not_in_finally(self):
        repo = Path(__file__).resolve().parents[1]
        src = (repo / "src" / "mad_mario" / "training" / "trainer.py").read_text(encoding="utf-8")
        save_idx = src.index("model.save(final_path)")
        finally_idx = src.index("finally:")
        complete_idx = src.index('logger.info("训练完成。")')
        assert save_idx < complete_idx < finally_idx
