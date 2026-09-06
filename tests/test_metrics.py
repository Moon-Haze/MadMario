"""测试 logging/metrics.py——指标记录器和回调辅助方法。"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from mad_mario.logging.metrics import METRIC_DECIMALS, MarioMetricsCallback


class TestMeanLast:
    """测试 MarioMetricsCallback._mean_last()。"""

    def test_empty(self):
        callback = MarioMetricsCallback(save_dir=Path("/tmp"))
        assert callback._mean_last([], as_int=False) == 0

    def test_single(self):
        callback = MarioMetricsCallback(save_dir=Path("/tmp"))
        assert callback._mean_last([42.0]) == 42.0

    def test_multiple(self):
        callback = MarioMetricsCallback(save_dir=Path("/tmp"))
        assert callback._mean_last([1.0, 2.0, 3.0]) == 2.0

    def test_as_int(self):
        callback = MarioMetricsCallback(save_dir=Path("/tmp"))
        result = callback._mean_last([1.2, 2.8, 3.5], as_int=True)
        assert isinstance(result, int)
        # mean=2.5, round(2.5) → 2（Python banker's rounding）
        assert result == 2

    def test_window(self):
        """只取最近 100 个值。"""
        callback = MarioMetricsCallback(save_dir=Path("/tmp"))
        values = list(range(200))
        # 最近 100 个：100..199，均值 149.5
        expected = round(float(sum(range(100, 200))) / 100, METRIC_DECIMALS)
        assert callback._mean_last(values) == expected


class TestCsvHeader:
    def test_header_written(self, tmp_path: Path):
        """首次创建 CSV 时应写入表头。"""
        callback = MarioMetricsCallback(save_dir=tmp_path, log_interval=100)
        # 手动触发表头写入
        callback.csv_path = tmp_path / "latest_metrics.csv"
        callback._ensure_csv_header()
        assert callback.csv_path.exists()
        with open(callback.csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "episode" in header
        assert "mean_reward" in header
        assert "mean_max_x_pos" in header

    def test_header_not_duplicated(self, tmp_path: Path):
        """文件已存在且有内容时不应重复写表头。"""
        callback = MarioMetricsCallback(save_dir=tmp_path, log_interval=100)
        callback.csv_path = tmp_path / "latest_metrics.csv"

        # 第一次写入
        callback._ensure_csv_header()
        # 第二次调用不应重复写
        callback._ensure_csv_header()

        with open(callback.csv_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1  # 只有一行表头

    def test_append_csv(self, tmp_path: Path):
        """CSV 追加写入应正确。"""
        callback = MarioMetricsCallback(save_dir=tmp_path, log_interval=100)
        callback.csv_path = tmp_path / "latest_metrics.csv"
        callback._ensure_csv_header()
        callback._append_csv([1, 100, 0.1, 50.0, 10, 0.0, 0.0, 300.0, 0.0, 0, 1.5])

        with open(callback.csv_path, "r") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2  # 表头 + 数据行
        assert rows[1][0] == "1"
        assert rows[1][3] == "50.0"


class TestRecordLogging:
    def test_record_emits_text_via_logging(self, tmp_path: Path, caplog):
        callback = MarioMetricsCallback(save_dir=tmp_path, log_interval=20)
        callback.csv_path = tmp_path / "latest_metrics.csv"
        callback._ensure_csv_header()
        callback._write_plots = lambda: None  # 避免真实绘图
        callback.model = SimpleNamespace(exploration_rate=0.5)
        callback.num_timesteps = 1000
        callback.episode_count = 20
        callback.ep_rewards = [10.0, 20.0]
        with caplog.at_level(logging.INFO, logger="mad_mario.logging.metrics"):
            callback._record()
        assert "回合=20" in caplog.text
        assert "平均奖励=" in caplog.text
        assert "平均最大 x=" in caplog.text
