"""测试 logging/metrics.py——指标记录器和回调辅助方法。"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mad_mario.logging.metrics import (
    MAX_PLOT_POINTS,
    METRIC_DECIMALS,
    Q_SAMPLE_SIZE,
    MarioMetricsCallback,
    _downsample,
)

CSV_HEADER = [
    "episode", "step", "epsilon",
    "mean_reward", "mean_length", "mean_loss", "mean_q",
    "mean_max_x_pos", "mean_noop_rate", "mean_most_used_action",
    "time_delta",
]


def _write_csv(csv_path: Path, rows: list[list]):
    """按真实表头写出一个历史 CSV。"""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


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


class TestHistoryContinuation:
    """断点重启后，图表必须接着上一次的历史继续画。"""

    def test_restores_history_from_existing_csv(self, tmp_path: Path):
        """启动时应把 CSV 里已有的曲线历史读回内存。"""
        csv_path = tmp_path / "latest_metrics.csv"
        _write_csv(csv_path, [
            [20, 1920, 1.0, 39.475, 40.65, 0.0, 0.0, 357.35, 1.0, 0, 1.053],
            [40, 3144, 1.0, 48.224, 49.875, 0.0, 0.0, 407.275, 1.0, 0, 1.443],
        ])
        callback = MarioMetricsCallback(save_dir=tmp_path, log_interval=20)
        callback.csv_path = csv_path

        callback._load_history_from_csv()

        assert callback.moving_avgs[0] == [39.475, 48.224]
        assert callback.moving_avgs[1] == [40.65, 49.875]
        assert callback.moving_avgs[2] == [0.0, 0.0]
        assert callback.moving_avgs[3] == [0.0, 0.0]

    def test_training_start_hook_restores_history(self, tmp_path: Path):
        """``_on_training_start`` 是接续的接线点，必须真的触发回读。"""
        csv_path = tmp_path / "latest_metrics.csv"
        _write_csv(csv_path, [
            [20, 1920, 1.0, 39.475, 40.65, 0.0, 0.0, 357.35, 1.0, 0, 1.053],
        ])
        callback = MarioMetricsCallback(save_dir=tmp_path, log_interval=20)
        # training_env 是 BaseCallback 的只读 property，从 model.get_env() 取
        callback.model = SimpleNamespace(
            action_space=SimpleNamespace(n=5),
            get_env=lambda: SimpleNamespace(num_envs=2),
        )

        callback._on_training_start()

        assert callback.moving_avgs[0] == [39.475], "训练启动后必须已带上历史"

    def test_new_record_continues_after_history(self, tmp_path: Path):
        """新记录点应追加在历史之后，而不是把历史顶掉。"""
        csv_path = tmp_path / "latest_metrics.csv"
        _write_csv(csv_path, [
            [20, 1920, 1.0, 39.475, 40.65, 0.0, 0.0, 357.35, 1.0, 0, 1.053],
        ])
        callback = MarioMetricsCallback(save_dir=tmp_path, log_interval=20)
        callback.csv_path = csv_path
        callback._load_history_from_csv()
        callback._write_plots = lambda: None   # 避免真实绘图
        callback.model = SimpleNamespace(exploration_rate=0.5)
        callback.num_timesteps = 5000
        callback.episode_count = 20
        callback.ep_rewards = [10.0, 20.0]

        callback._record()

        assert len(callback.moving_avgs[0]) == 2, "应接在历史之后"
        assert callback.moving_avgs[0][0] == 39.475, "历史点不能被覆盖"

    def test_skips_unparseable_rows(self, tmp_path: Path):
        """列缺失或数值非法的行应跳过，不影响其余历史。"""
        csv_path = tmp_path / "latest_metrics.csv"
        _write_csv(csv_path, [
            [20, 1920, 1.0, 39.475, 40.65, 0.0, 0.0, 1.0, 1.0, 0, 1.0],
            [40, 3144, 1.0, "坏值", 40.65, 0.0, 0.0, 1.0, 1.0, 0, 1.0],
            [60, 5000],   # 半行——训练被强杀时可能留下
            [80, 6000, 1.0, 50.0, 42.0, 0.0, 0.0, 1.0, 1.0, 0, 1.0],
        ])
        callback = MarioMetricsCallback(save_dir=tmp_path, log_interval=20)
        callback.csv_path = csv_path

        callback._load_history_from_csv()

        assert callback.moving_avgs[0] == [39.475, 50.0], "只保留可解析的行"

    def test_absent_csv_is_noop(self, tmp_path: Path):
        """CSV 尚不存在时应安静返回，不留半个状态。"""
        callback = MarioMetricsCallback(save_dir=tmp_path, log_interval=20)
        callback.csv_path = tmp_path / "latest_metrics.csv"

        callback._load_history_from_csv()

        assert callback.moving_avgs == [[], [], [], []]


class TestDownsample:
    """绘图抽稀：点数少时原样画，点数多时等距抽稀但保留首尾。"""

    def test_short_series_untouched(self):
        xs, ys = _downsample([1.0, 2.0, 3.0])
        assert list(xs) == [1, 2, 3]
        assert list(ys) == [1.0, 2.0, 3.0]

    def test_empty_series(self):
        xs, ys = _downsample([])
        assert len(xs) == 0
        assert len(ys) == 0

    def test_long_series_thinned_keeping_endpoints(self):
        values = [float(i) for i in range(MAX_PLOT_POINTS * 3)]
        xs, ys = _downsample(values)

        assert len(xs) <= MAX_PLOT_POINTS
        assert len(xs) == len(ys)
        assert xs[0] == 1, "首点必须是第 1 个记录点"
        assert xs[-1] == len(values), "末点必须保留，否则曲线尾部被截断"
        assert all(b > a for a, b in zip(xs, xs[1:])), "x 轴必须严格递增"


def _make_callback(
    tmp_path: Path,
    n_envs: int = 2,
    n_actions: int = 5,
) -> MarioMetricsCallback:
    """构造一个已完成训练启动初始化的回调。

    ``_act_count`` 等按环境数分配的状态只在 ``_on_training_start`` 里建立，
    因此动作类测试必须先走一遍该钩子。
    """
    callback = MarioMetricsCallback(save_dir=tmp_path, log_interval=100)
    callback.model = SimpleNamespace(
        action_space=SimpleNamespace(n=n_actions),
        get_env=lambda: SimpleNamespace(num_envs=n_envs),
        # 真实 model 是 DQN，logger 由 SB3 注入；这里给个空表即可
        logger=SimpleNamespace(name_to_value={}),
    )
    callback._on_training_start()
    return callback


class TestStepActionStats:
    """动作统计必须来自 SB3 真实下发的动作，而不是退化成常量。"""

    def test_counts_actions_from_locals(self, tmp_path: Path):
        """动作应从 ``locals["actions"]`` 取，而非恒为 0 号。"""
        callback = _make_callback(tmp_path)
        callback.locals = {
            "actions": np.array([2, 4]),
            "rewards": np.array([1.0, 1.0], dtype=np.float32),
            # 回合不结束，计数才会留在累积器里（结束时会清零）
            "dones": np.array([False, False]),
            "infos": [{}, {}],
        }

        callback._on_step()

        assert callback._act_count[0][2] == 1, "env 0 收到的是动作 2"
        assert callback._act_count[1][4] == 1, "env 1 收到的是动作 4"

    def test_episode_most_used_action_is_not_always_zero(self, tmp_path: Path):
        """回合记录的最常用动作应反映真实动作，而非恒定 0。"""
        callback = _make_callback(tmp_path)
        callback.locals = {
            "actions": np.array([2, 2]),
            "rewards": np.array([0.0, 0.0], dtype=np.float32),
            "dones": np.array([True, False]),
            "infos": [{}, {}],
        }

        callback._on_step()

        assert callback.ep_most_used_actions == [2]

    def test_noop_rate_only_for_action_zero(self, tmp_path: Path):
        """只有 0 号动作才算 NOOP。"""
        callback = _make_callback(tmp_path)
        callback.locals = {
            "actions": np.array([0, 3]),
            "rewards": np.array([0.0, 0.0], dtype=np.float32),
            "dones": np.array([True, True]),
            "infos": [{}, {}],
        }

        callback._on_step()

        assert callback.ep_noop_rates == [1.0, 0.0]

    def test_missing_actions_key_degrades_to_noop(self, tmp_path: Path):
        """拿不到动作时应安静退化为 NOOP，而不是抛异常中断训练。"""
        callback = _make_callback(tmp_path)
        callback.locals = {
            "rewards": np.array([0.0], dtype=np.float32),
            "dones": np.array([True]),
            "infos": [{}],
        }

        callback._on_step()

        assert callback.ep_noop_rates == [1.0]

    def test_out_of_range_action_ignored(self, tmp_path: Path):
        """动作越界时只跳过计数，不得让训练崩溃。"""
        callback = _make_callback(tmp_path, n_actions=5)
        callback.locals = {
            "actions": np.array([99]),
            "rewards": np.array([0.0], dtype=np.float32),
            "dones": np.array([True]),
            "infos": [{}],
        }

        callback._on_step()

        assert callback._act_count[0].tolist() == [0, 0, 0, 0, 0]


class TestEstimateQ:
    """Q 值曲线必须有真实来源，而不是写死的 0。"""

    def test_returns_zero_without_replay_buffer(self, tmp_path: Path):
        callback = MarioMetricsCallback(save_dir=tmp_path)
        callback.model = SimpleNamespace()

        assert callback._estimate_q() == 0.0

    def test_returns_zero_when_buffer_too_small(self, tmp_path: Path):
        """样本数不足采样数时无法估计，返回 0.0。"""
        callback = MarioMetricsCallback(save_dir=tmp_path)
        callback.model = SimpleNamespace(
            replay_buffer=SimpleNamespace(size=lambda: Q_SAMPLE_SIZE - 1),
        )

        assert callback._estimate_q() == 0.0

    def test_returns_mean_of_max_q(self, tmp_path: Path):
        """结果为各状态最大 Q 的均值，且采样到的观测确实被喂给 q_net。"""
        seen: list = []

        class _FakeQNet:
            def __call__(self, observations):
                seen.append(observations)
                return torch.tensor([[1.0, 3.0], [5.0, 2.0]])

        callback = MarioMetricsCallback(save_dir=tmp_path)
        callback.model = SimpleNamespace(
            replay_buffer=SimpleNamespace(
                size=lambda: Q_SAMPLE_SIZE,
                sample=lambda _n: SimpleNamespace(observations="观测张量"),
            ),
            q_net=_FakeQNet(),
        )

        assert callback._estimate_q() == 4.0, "max Q 依次为 3、5，均值应为 4"
        assert seen == ["观测张量"], "必须把采样到的观测喂给 q_net"


class TestEpisodeQ:
    """回合记录的 Q 值应来自估计，而不是写死的 0。"""

    def test_episode_records_estimated_q(self, tmp_path: Path):
        callback = _make_callback(tmp_path)
        callback._estimate_q = lambda: 12.5
        callback.locals = {
            "actions": np.array([1, 1]),
            "rewards": np.array([0.0, 0.0], dtype=np.float32),
            "dones": np.array([True, False]),
            "infos": [{}, {}],
        }

        callback._on_step()

        assert callback.ep_avg_qs == [12.5]

    def test_q_estimated_once_per_step(self, tmp_path: Path):
        """同一步多个环境同时结束回合时，Q 估计只做一次。"""
        calls: list = []

        def _fake_estimate() -> float:
            calls.append(1)
            return 1.0

        callback = _make_callback(tmp_path)
        callback._estimate_q = _fake_estimate
        callback.locals = {
            "actions": np.array([1, 1]),
            "rewards": np.array([0.0, 0.0], dtype=np.float32),
            "dones": np.array([True, True]),
            "infos": [{}, {}],
        }

        callback._on_step()

        assert len(calls) == 1, "两个回合同时结束不应重复前向"
        assert callback.ep_avg_qs == [1.0, 1.0]
