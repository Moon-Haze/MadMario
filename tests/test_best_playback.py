"""测试训练中最优模型刷新时的自动录制（training/best_playback.py）。"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.callbacks import EvalCallback

from mad_mario.config import AppConfig, EnvConfig, PlayConfig, SB3ArtifactConfig, SB3Config
from mad_mario.training import best_playback
from mad_mario.training.best_playback import (
    BEST_VIDEO_DIRNAME,
    BestModelPlaybackCallback,
    _clears_the_bar,
    _to_playback_config,
)


def _train_config(tmp_path: Path, *, play_on_best: bool = True) -> AppConfig:
    """造一份训练配置，形状与 config_from_train_args 的产出一致。"""
    return AppConfig(
        env=EnvConfig(levels=["SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0"]),
        sb3=SB3Config(),
        artifacts=SB3ArtifactConfig(save_root=tmp_path),
        play=PlayConfig(video_fps=30, play_on_best=play_on_best),
    )


class _FakeModel:
    """只提供录制链路真正会碰的两样东西：一个步数，和一个会落盘的 save()。"""

    num_timesteps = 0

    def save(self, path):
        Path(path).write_bytes(b"fake best model")


class _PlaySpy:
    """替掉 best_playback.play：记下调用，并可选地卡住，模拟一趟耗时很长的录制。

    现在是后台线程在调它，所以记录要用锁护住；gate 用来把"录制还在跑"这个
    瞬间稳定地摆出来，否则测试只能靠 sleep 撞运气。
    """

    def __init__(self, error: Exception | None = None, gate: threading.Event | None = None,
                 entered: threading.Event | None = None):
        self.calls: list[AppConfig] = []
        self.kwargs: list[dict] = []
        self._error = error
        self._gate = gate
        self._entered = entered
        self._lock = threading.Lock()

    def __call__(self, config, **kwargs):
        with self._lock:
            self.calls.append(config)
            self.kwargs.append(kwargs)
        if self._entered is not None:
            self._entered.set()
        if self._gate is not None:
            # 等锁放开之后再等，否则后台线程会抱着锁睡死，测试连读都读不了
            self._gate.wait(timeout=5)
        if self._error is not None:
            raise self._error

    def video_dirs(self) -> list[str]:
        """按录制顺序列出每一趟落到的目录名。"""
        with self._lock:
            return [call.play.video_dir.name for call in self.calls]

    def checkpoints(self) -> list[Path]:
        with self._lock:
            return [call.artifacts.checkpoint for call in self.calls]


@pytest.fixture
def _no_op_super_init(monkeypatch):
    """把 EvalCallback.__init__ 换成桩。

    父类构造要求一个真的 VecEnv，还会做一次评估；录制逻辑全在 _on_step 与
    存档/录制链路上，测它没必要付这份代价。桩里补上父类 __init__ 本来会设的
    那几个属性（CallbackList 每步刷新的 num_timesteps、初值 -inf 的
    best_mean_reward、以及存档要用的 model），语义与真实一致。
    """
    def _stub_init(self, *args, **kwargs):
        self.model = _FakeModel()
        self.num_timesteps = 0
        self.best_mean_reward = -np.inf

    monkeypatch.setattr(EvalCallback, "__init__", _stub_init)


def _make_callback(monkeypatch, tmp_path: Path, *, play_on_best: bool = True,
                   steps: int = 160_000, min_improvement: float = 0.05):
    """构造一个可用的回调：父类初始化打桩，其余走真实代码。"""
    config = _train_config(tmp_path, play_on_best=play_on_best)
    callback = BestModelPlaybackCallback(
        train_config=config,
        best_model_save_path=tmp_path / "best_model",
        min_improvement=min_improvement,
    )
    callback.num_timesteps = steps
    return callback, config


def _refresh_best_to(value: float):
    """替掉父类 _on_step：模拟"本次评估刷新了最优成绩"这一时机。"""
    def _on_step(self):
        self.best_mean_reward = value
        return True

    return _on_step


class TestToPlaybackConfig:
    """训练配置 → 播放配置的改造：只录制、不塑形、不弹窗。"""

    def test_points_at_the_best_model(self, tmp_path: Path):
        best = tmp_path / "best_model" / "best_model.zip"
        config = _to_playback_config(_train_config(tmp_path), best, tmp_path / "videos")

        assert config.artifacts.checkpoint == best

    def test_never_shows_a_window(self, tmp_path: Path):
        """训练途中弹窗会打断使用者，也可能根本没有显示器。"""
        config = _to_playback_config(_train_config(tmp_path), tmp_path / "m.zip", tmp_path)

        assert config.play.show is False

    def test_keeps_configured_fps(self, tmp_path: Path):
        config = _to_playback_config(_train_config(tmp_path), tmp_path / "m.zip", tmp_path)

        assert config.play.video_fps == 30

    def test_skips_the_timestamp_dir(self, tmp_path: Path):
        """step_xxx 已经分好档了，再套时间戳只是套娃，还会让路径白长一截。"""
        config = _to_playback_config(_train_config(tmp_path), tmp_path / "m.zip", tmp_path)

        assert config.play.timestamp_videos is False

    def test_keeps_greedy_actions(self, tmp_path: Path):
        """训练侧录的是"这个最优模型的最好表现"，不能被采样噪声糊掉。"""
        config = _to_playback_config(_train_config(tmp_path), tmp_path / "m.zip", tmp_path)

        assert config.play.sample_runs == 0, "0 遍采样 = 只贪心录一遍"

    def test_writes_to_given_video_dir(self, tmp_path: Path):
        video_dir = tmp_path / "nested" / "videos"
        config = _to_playback_config(_train_config(tmp_path), tmp_path / "m.zip", video_dir)

        assert config.play.video_dir == video_dir

    def test_disables_reward_shaping(self, tmp_path: Path):
        """播放是观赏不是训练：塑形奖励与卡住惩罚都不该再生效。"""
        train_config = _train_config(tmp_path)
        assert train_config.env.progress_reward_enabled, "前提：训练配置默认是开着的"

        config = _to_playback_config(train_config, tmp_path / "m.zip", tmp_path)

        assert config.env.clip_rewards is False
        assert config.env.stuck_penalty_enabled is False
        assert config.env.progress_reward_enabled is False
        assert config.env.score_reward_enabled is False
        assert config.env.flag_reward_enabled is False

    def test_keeps_stuck_rescue(self, tmp_path: Path):
        """脱困挣扎要留着，否则模型卡住时画面会一直僵着。"""
        config = _to_playback_config(_train_config(tmp_path), tmp_path / "m.zip", tmp_path)

        assert config.env.stuck_rescue_enabled is True

    def test_records_every_training_level(self, tmp_path: Path):
        """录的是训练用的全部关卡，不是单关。"""
        config = _to_playback_config(_train_config(tmp_path), tmp_path / "m.zip", tmp_path)

        assert config.env.levels == ["SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0"]


class TestRefreshBar:
    """刷新门槛：评估噪声的每个小波峰都不该换来一次存档加录制。"""

    def test_first_best_always_qualifies(self):
        """还没有基准可比时，任何成绩都是第一个最优。"""
        assert _clears_the_bar(-np.inf, -50.0, 0.05) is True

    def test_small_gain_is_rejected(self):
        assert _clears_the_bar(100.0, 103.0, 0.05) is False

    def test_gain_past_the_bar_is_accepted(self):
        assert _clears_the_bar(100.0, 105.0, 0.05) is True

    def test_bar_works_with_negative_rewards(self):
        """奖励为负时门槛不能反着往下滑：-97 比 -100 好，但还差得远。"""
        assert _clears_the_bar(-100.0, -97.0, 0.05) is False
        assert _clears_the_bar(-100.0, -90.0, 0.05) is True

    def test_zero_threshold_restores_old_behaviour(self):
        """门槛设 0 就退回父类"高一点算一点"的老行为。"""
        assert _clears_the_bar(100.0, 100.1, 0.0) is True


class TestBestModelPlaybackCallback:
    """刷新检测、门槛判定与录制时机：录制失败绝不能把训练拖垮。"""

    def test_records_when_best_refreshes(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path, steps=160_000)
        assert callback._on_step() is True
        callback.wait_for_recording()

        assert len(play_spy.calls) == 1, "刷新一次就该录一次"

    def test_video_dir_is_step_stamped_and_nested_under_save_root(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """按步数分目录，多次刷新的视频才能按训练阶段对比着看。"""
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path, steps=160_000)
        callback._on_step()
        callback.wait_for_recording()

        expected = tmp_path / BEST_VIDEO_DIRNAME / "step_000160000"
        assert play_spy.calls[0].play.video_dir == expected

    def test_records_a_frozen_copy_of_the_best_model(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """录的必须是当时那份模型：best_model.zip 会被后来的刷新原地覆盖。

        录制排在后台队列里，等轮到它时源文件可能已经是更晚的模型了——目录名
        写着 step_000160000，录出来却是 step_000210000 的表现。
        """
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path)
        callback._on_step()
        callback.wait_for_recording()

        recorded = play_spy.checkpoints()[0]
        assert recorded == tmp_path / BEST_VIDEO_DIRNAME / "step_000160000" / "best_model.zip"
        assert recorded.exists(), "快照要真的落盘，后台那一趟才有东西可加载"

    def test_saves_the_best_model_itself(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """门槛判定收在回调手里，存档也跟着搬过来——够格才写盘。"""
        monkeypatch.setattr(best_playback, "play", _PlaySpy())
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path)
        callback._on_step()
        callback.wait_for_recording()

        assert (tmp_path / "best_model" / "best_model.zip").exists()

    def test_does_not_swallow_training_interrupt(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """后台线程收不到信号，吞不吞中断都一样：不能替主线程把停止咽下去。"""
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path)
        callback._on_step()
        callback.wait_for_recording()

        assert play_spy.kwargs[0]["catch_interrupt"] is False

    def test_skips_when_best_unchanged(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """评估没变好就不该录，否则每次评估都要白白跑一遍全部关卡。"""
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        monkeypatch.setattr(EvalCallback, "_on_step", lambda self: True)

        callback, _ = _make_callback(monkeypatch, tmp_path)
        assert callback._on_step() is True

        assert play_spy.calls == []

    def test_skips_gains_below_the_bar(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """高一点点不算刷新：否则评估噪声的每个小波峰都要录一遍。"""
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        callback, _ = _make_callback(monkeypatch, tmp_path, min_improvement=0.5)

        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(100.0))
        callback._on_step()
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(120.0))
        callback._on_step()
        callback.wait_for_recording()

        assert play_spy.video_dirs() == ["step_000160000"], "只有第一次（无基准）算数"

    def test_rejected_gains_do_not_drift_the_bar(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """门槛比的是已采纳的最优，不是上一次评估。

        100 是首次，没有基准可比，必然采纳；之后 110、120 都是噪声级的小涨，
        够不着 100×1.5 的门槛，都不该采纳。而被拒时若不把标准退回去，
        best_mean_reward 会一路跟着噪声爬到 120，于是 160 这次真实提升就够不着
        120×1.5=180，被白白漏掉——那样只会录下首趟，第二趟不会出现。
        """
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        callback, _ = _make_callback(monkeypatch, tmp_path, min_improvement=0.5)

        for reward in (100.0, 110.0, 120.0, 160.0):
            monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(reward))
            callback._on_step()

        callback.wait_for_recording()
        assert len(play_spy.calls) == 2, "首次那趟加上 160 那趟；中间的小涨都不算"
        assert callback.best_mean_reward == 160.0

    def test_respects_play_on_best_false(self, tmp_path: Path, monkeypatch, _no_op_super_init):
        """--no-play-best 时评估照常，只是不录制。"""
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path, play_on_best=False)
        assert callback._on_step() is True, "关掉录制不该影响训练继续"

        assert play_spy.calls == []

    def test_recording_failure_does_not_break_training(
        self, tmp_path: Path, monkeypatch, _no_op_super_init, caplog
    ):
        """录制是观察手段：失败只告警，训练必须继续。"""
        monkeypatch.setattr(best_playback, "play", _PlaySpy(error=RuntimeError("磁盘满了")))
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path)
        with caplog.at_level("WARNING"):
            assert callback._on_step() is True
            callback.wait_for_recording()

        assert "磁盘满了" in caplog.text, "失败原因要留在日志里，否则没人知道视频缺了"

    def test_failure_does_not_drop_the_next_recording(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """一趟坏掉不该让排在后面的任务跟着陪葬。"""
        entered, gate = threading.Event(), threading.Event()
        failing = _PlaySpy(error=RuntimeError("环境炸了"), gate=gate, entered=entered)
        monkeypatch.setattr(best_playback, "play", failing)
        callback, _ = _make_callback(monkeypatch, tmp_path)

        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))
        callback.num_timesteps = 160_000
        callback._on_step()
        assert entered.wait(timeout=5), "前提：第一趟已经进了 play"

        # 第二趟换个不炸的 spy；worker 每次现查模块全局，换掉即生效
        healthy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", healthy)
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(300.0))
        callback.num_timesteps = 210_000
        callback._on_step()

        gate.set()
        callback.wait_for_recording()

        assert healthy.video_dirs() == ["step_000210000"], "前一趟炸了，后一趟照样要录"

    def test_propagates_stop_request(self, tmp_path: Path, monkeypatch, _no_op_super_init):
        """父类喊停时必须原样返回 False，不能被录制的返回值盖掉。"""
        monkeypatch.setattr(best_playback, "play", _PlaySpy())
        monkeypatch.setattr(EvalCallback, "_on_step", lambda self: False)

        callback, _ = _make_callback(monkeypatch, tmp_path)

        assert callback._on_step() is False

    def test_video_root_lives_under_save_root(self, tmp_path: Path, monkeypatch, _no_op_super_init):
        callback, _ = _make_callback(monkeypatch, tmp_path)

        assert callback._video_root == tmp_path / BEST_VIDEO_DIRNAME


class TestBackgroundRecording:
    """后台录制：训练主循环不被挡住，收尾时又能把已排队的视频都收干净。"""

    def test_recording_does_not_block_training(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """一趟录制要跑遍全部关卡，卡住主循环等于把评估间隔拉长好几倍。"""
        gate = threading.Event()
        monkeypatch.setattr(best_playback, "play", _PlaySpy(gate=gate))
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path)
        # 后台那一趟正卡在 play 里；_on_step 必须已经返回，否则这里就挂住了
        started = time.monotonic()
        assert callback._on_step() is True
        elapsed = time.monotonic() - started

        gate.set()
        callback.wait_for_recording()

        assert elapsed < 1.0, f"录制把训练主循环挡了 {elapsed:.1f} 秒"

    def test_every_refresh_gets_recorded_in_order(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """排队逐个录：连着刷新两次，两档都不能丢。"""
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        callback, _ = _make_callback(monkeypatch, tmp_path)

        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))
        callback.num_timesteps = 160_000
        callback._on_step()

        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(300.0))
        callback.num_timesteps = 210_000
        callback._on_step()

        callback.wait_for_recording()

        assert play_spy.video_dirs() == ["step_000160000", "step_000210000"]

    def test_training_end_waits_for_pending_recording(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """收尾要等队列清空：带着写了一半的 mp4 退出，播放器连打开都打不开。"""
        gate = threading.Event()
        play_spy = _PlaySpy(gate=gate)
        monkeypatch.setattr(best_playback, "play", play_spy)
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path)
        callback._on_step()

        # 后台还卡在录制里；收尾若不等，返回时线程必然还活着
        timer = threading.Timer(0.2, gate.set)
        timer.start()
        try:
            callback._on_training_end()
        finally:
            timer.cancel()

        assert not callback._worker.is_alive(), "收尾返回时后台线程必须已经退出"
        assert len(play_spy.calls) == 1

    def test_worker_is_not_a_daemon(self, tmp_path: Path, monkeypatch, _no_op_super_init):
        """守护线程会在解释器退出时被当场掐断，留下打不开的半截 mp4。"""
        monkeypatch.setattr(best_playback, "play", _PlaySpy())
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path)
        callback._on_step()
        callback.wait_for_recording()

        assert callback._worker.daemon is False

    def test_no_worker_thread_when_nothing_is_recorded(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """没开录制就不该白养一个线程。"""
        monkeypatch.setattr(best_playback, "play", _PlaySpy())
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path, play_on_best=False)
        callback._on_step()
        callback.wait_for_recording()

        assert callback._worker is None

    def test_wait_is_idempotent(self, tmp_path: Path, monkeypatch, _no_op_super_init):
        """正常结束与 Ctrl+C 都会走到等待，重复调用不能挂死或报错。"""
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path)
        callback._on_step()

        callback.wait_for_recording()
        callback.wait_for_recording()
        callback._on_training_end()

        assert len(play_spy.calls) == 1, "重复等待不该把同一趟录两遍"
