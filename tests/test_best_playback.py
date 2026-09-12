"""测试训练中最优模型刷新时的自动录制（training/best_playback.py）。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3.common.callbacks import EvalCallback

from mad_mario.config import AppConfig, EnvConfig, PlayConfig, SB3ArtifactConfig, SB3Config
from mad_mario.training import best_playback
from mad_mario.training.best_playback import (
    BEST_VIDEO_DIRNAME,
    BestModelPlaybackCallback,
    _to_playback_config,
)


def _train_config(tmp_path: Path, *, play_on_best: bool = True, steps: int = 160_000):
    """造一份训练配置，形状与 config_from_train_args 的产出一致。"""
    return AppConfig(
        env=EnvConfig(levels=["SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0"]),
        sb3=SB3Config(),
        artifacts=SB3ArtifactConfig(save_root=tmp_path),
        play=PlayConfig(video_fps=30, play_on_best=play_on_best),
    )


class _PlaySpy:
    """替掉 best_playback.play，只记下被调用时的配置，不真去跑环境。"""

    def __init__(self, error: Exception | None = None):
        self.calls: list[AppConfig] = []
        self.kwargs: list[dict] = []
        self._error = error

    def __call__(self, config, **kwargs):
        self.calls.append(config)
        self.kwargs.append(kwargs)
        if self._error is not None:
            raise self._error


@pytest.fixture
def _no_op_super_init(monkeypatch):
    """把 EvalCallback.__init__ 换成桩。

    父类构造要求一个真的 VecEnv，还会做一次评估；录制逻辑全在 _on_step 与
    _record_best 里，测它没必要付这份代价。桩里补上父类 __init__ 本来会设的
    那几个属性（CallbackList 每步刷新的 num_timesteps、初值 -inf 的
    best_mean_reward、以及供 num_timesteps 兜底的 model），语义与真实一致。
    """
    def _stub_init(self, *args, **kwargs):
        self.model = SimpleNamespace(num_timesteps=0)
        self.num_timesteps = 0
        self.best_mean_reward = -np.inf

    monkeypatch.setattr(EvalCallback, "__init__", _stub_init)


def _make_callback(monkeypatch, tmp_path: Path, *, play_on_best: bool = True, steps: int = 160_000):
    """构造一个可用的回调：父类初始化打桩，其余走真实代码。"""
    config = _train_config(tmp_path, play_on_best=play_on_best)
    callback = BestModelPlaybackCallback(train_config=config)
    callback.num_timesteps = steps
    callback.best_model_save_path = str(tmp_path / "best_model")
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


class TestBestModelPlaybackCallback:
    """刷新检测与录制时机：录制失败绝不能把训练拖垮。"""

    def test_records_when_best_refreshes(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path, steps=160_000)
        assert callback._on_step() is True

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

        expected = tmp_path / BEST_VIDEO_DIRNAME / "step_000160000"
        assert play_spy.calls[0].play.video_dir == expected

    def test_records_the_best_model_file(self, tmp_path: Path, monkeypatch, _no_op_super_init):
        """录的必须是刚存下的 best_model.zip，而不是别的 checkpoint。"""
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path)
        callback._on_step()

        assert play_spy.calls[0].artifacts.checkpoint == tmp_path / "best_model" / "best_model.zip"

    def test_does_not_swallow_training_interrupt(
        self, tmp_path: Path, monkeypatch, _no_op_super_init
    ):
        """嵌套调用时 Ctrl+C 必须照常冒泡，否则使用者以为停止没生效。"""
        play_spy = _PlaySpy()
        monkeypatch.setattr(best_playback, "play", play_spy)
        monkeypatch.setattr(EvalCallback, "_on_step", _refresh_best_to(200.0))

        callback, _ = _make_callback(monkeypatch, tmp_path)
        callback._on_step()

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

        assert "磁盘满了" in caplog.text, "失败原因要留在日志里，否则没人知道视频缺了"

    def test_propagates_stop_request(self, tmp_path: Path, monkeypatch, _no_op_super_init):
        """父类喊停时必须原样返回 False，不能被录制的返回值盖掉。"""
        monkeypatch.setattr(best_playback, "play", _PlaySpy())
        monkeypatch.setattr(EvalCallback, "_on_step", lambda self: False)

        callback, _ = _make_callback(monkeypatch, tmp_path)

        assert callback._on_step() is False

    def test_video_root_lives_under_save_root(self, tmp_path: Path, monkeypatch, _no_op_super_init):
        callback, _ = _make_callback(monkeypatch, tmp_path)

        assert callback._video_root == tmp_path / BEST_VIDEO_DIRNAME
