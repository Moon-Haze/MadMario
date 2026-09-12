"""测试 playback/video.py 的录制器，以及 playback.runner.play 的录制接线。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as iio
import numpy as np
import pytest

from mad_mario.playback.runner import _unique_path, _video_filename
from mad_mario.playback.video import VideoRecorder


def _frames(count: int, size: int = 64) -> list[np.ndarray]:
    """生成 count 个可区分的纯色帧。"""
    return [
        np.full((size, size, 3), fill_value=(i * 10) % 256, dtype=np.uint8)
        for i in range(count)
    ]


class TestVideoRecorder:
    """录制器本身。真编一遍 mp4，确认产出文件可读且帧数对得上。"""

    def test_writes_playable_mp4(self, tmp_path: Path):
        path = tmp_path / "out.mp4"
        with VideoRecorder(path, fps=15) as recorder:
            for frame in _frames(5):
                recorder.append(frame)

        assert path.exists(), "应产出视频文件"
        reader = iio.get_reader(path)
        try:
            meta = reader.get_meta_data()
            count = reader.count_frames()
        finally:
            reader.close()
        assert int(meta["fps"]) == 15, "帧率应写进容器"
        assert count == 5, "写入 5 帧就该读出 5 帧"

    def test_creates_parent_directory(self, tmp_path: Path):
        """父目录不存在时应自动创建，否则首次录制直接失败。"""
        path = tmp_path / "nested" / "deep" / "out.mp4"
        with VideoRecorder(path, fps=15) as recorder:
            recorder.append(_frames(1)[0])

        assert path.exists()

    def test_closes_writer_on_exception(self, tmp_path: Path):
        """异常路径下 writer 也必须关闭，否则泄漏 ffmpeg 子进程句柄。"""
        recorder = VideoRecorder(tmp_path / "out.mp4", fps=15)
        with pytest.raises(RuntimeError, match="关卡出错"):
            with recorder:
                recorder.append(_frames(1)[0])
                raise RuntimeError("关卡出错")

        assert recorder._writer is None

    def test_frame_count_tracks_appends(self, tmp_path: Path):
        with VideoRecorder(tmp_path / "out.mp4", fps=15) as recorder:
            for frame in _frames(3):
                recorder.append(frame)

        assert recorder.frame_count == 3


class _FakeModel:
    """只提供 play() 用到的那几个成员。

    deterministic_calls 记下每次推理传进来的 deterministic，用来验证
    --sample 确实改的是推理方式，而不只是改了个配置字段。

    sampled 记下采样分支抽到的随机数：真 DQN 的 ε-greedy 走的是全局 numpy
    随机流，这里照抄那条路径，才测得出"种子被按回固定值"这类问题。

    rates_used 记下每次推理时模型身上的 exploration_rate：play() 是改写这个
    属性来切换每遍的随机强度的，记下来才能验证文件名里的数字与实际用的是
    同一个值，而不是各写各的。
    """

    num_timesteps = 1234
    exploration_rate = 0.198

    def __init__(self):
        self.deterministic_calls: list[bool] = []
        self.sampled: list[float] = []
        self.rates_used: list[float] = []

    def predict(self, state, deterministic=True):
        self.deterministic_calls.append(deterministic)
        if not deterministic:
            self.sampled.append(float(np.random.rand()))
            self.rates_used.append(self.exploration_rate)
        return 1, None


class _FakeEnv:
    """最小可录制环境：跑满 n_steps 步后自然结束。

    interrupt_at 不为 None 时，在第该次 step 抛 KeyboardInterrupt，用于
    模拟录制途中按 Ctrl+C。
    """

    def __init__(self, n_steps: int = 3, interrupt_at: int | None = None):
        self.n_steps = n_steps
        self.interrupt_at = interrupt_at
        self.closed = False
        self.resets = 0
        self._step = 0

    def reset(self):
        self._step = 0
        self.resets += 1
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def step(self, action):
        self._step += 1
        if self.interrupt_at is not None and self._step >= self.interrupt_at:
            raise KeyboardInterrupt
        done = self._step >= self.n_steps
        return np.zeros((4, 84, 84), dtype=np.uint8), 1.0, done, False, {}

    def close(self):
        self.closed = True


class _FakeViewer:
    """假窗口：只记下被送了几帧、有没有被关上。

    默认 show=False 的测试用不到它；开启显示的测试靠它避开真实 matplotlib。
    close_after 不为 None 时，在第该次 show() 后把自己标记为已关闭，用来
    模拟用户中途关窗。
    """

    def __init__(self, fps: int = 15, close_after: int | None = None):
        self.fps = fps
        self.close_after = close_after
        self.shown = 0
        self.closed = False
        self.open = True

    def show(self, frame):
        self.shown += 1
        if self.close_after is not None and self.shown >= self.close_after:
            self.open = False

    @property
    def is_open(self):
        return self.open

    def close(self):
        self.closed = True


def _play_config(
    tmp_path: Path,
    levels: tuple[str, ...],
    video_dir: Path | None = None,
    show: bool = False,
    timestamp_videos: bool = False,
    sample_runs: int = 0,
):
    """构造 play() 需要的最小配置，并落下 checkpoint 文件让存在性检查通过。

    show 默认 False：录制相关的测试不该顺带拉起 matplotlib。
    timestamp_videos 默认 False：多数测试关心的是"哪个关卡落到哪个文件"，直接
    断言平铺路径最清楚；时间戳目录本身另有专门的测试覆盖。
    sample_runs 默认 0：贪心地跑一遍，与不传 --sample 时一致。
    """
    checkpoint = tmp_path / "model.zip"
    checkpoint.touch()
    return SimpleNamespace(
        env=SimpleNamespace(levels=list(levels), world=1, level=1),
        artifacts=SimpleNamespace(checkpoint=checkpoint, save_root=tmp_path),
        play=SimpleNamespace(
            video_dir=video_dir,
            video_fps=15,
            show=show,
            timestamp_videos=timestamp_videos,
            sample_runs=sample_runs,
        ),
    )


def _only_subdir(base: Path) -> Path:
    """取 base 下唯一的时间戳目录。

    多于一个说明这一场测试触发了不止一次播放，此时断言本就该失败。
    """
    subdirs = sorted(p for p in base.iterdir() if p.is_dir())
    assert len(subdirs) == 1, f"应恰好有一个时间戳目录，实际为 {subdirs}"
    return subdirs[0]


def _make_env_factory(
    env: _FakeEnv | None = None,
    envs: list[_FakeEnv] | None = None,
    render_modes: list | None = None,
):
    """造一个假环境工厂，顺带记录 play() 传入的 render_mode。

    play() 以 make_mario_env(config, render_mode=..., level_name=...) 调用，
    签名必须对得上，否则测试会以 TypeError 失败而不是断言失败。
    传 env 则每次返回同一个对象，否则每次新建一个。
    """

    def _factory(config, render_mode=None, level_name=None):
        if render_modes is not None:
            render_modes.append(render_mode)
        result = env if env is not None else _FakeEnv()
        if envs is not None:
            envs.append(result)
        return result

    return _factory


def _patch_play(monkeypatch, env_factory, model: _FakeModel | None = None):
    """把 play() 的模型加载与环境创建都换成假的。

    补丁打在 playback.runner 上而不是 cli 上：play() 实现住在 runner 里，
    cli 只是转发调用，改 cli 的属性影响不到真正的执行路径。
    传 model 可拿到同一个假模型实例，用于断言推理方式。
    """
    import mad_mario.playback.runner as runner_mod

    model = _FakeModel() if model is None else model
    monkeypatch.setattr(runner_mod, "DQN", SimpleNamespace(load=lambda *a, **k: model))
    monkeypatch.setattr(runner_mod, "make_mario_env", env_factory)
    return runner_mod


class TestPlayRecording:
    """play() 的录制接线：不碰真实环境、模型与显示器。"""

    def test_writes_one_video_per_level(self, tmp_path: Path, monkeypatch):
        envs: list[_FakeEnv] = []
        runner_mod = _patch_play(monkeypatch, _make_env_factory(envs=envs))
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0")
        )

        runner_mod.play(config)

        video_dir = tmp_path / "videos"
        assert (video_dir / "1-1.mp4").exists(), "每个关卡各一个文件"
        assert (video_dir / "1-2.mp4").exists()
        assert len(envs) == 2
        assert all(env.closed for env in envs), "每个关卡的环境都要关掉"

    def test_video_filename_uses_level_short_name(self, tmp_path: Path, monkeypatch):
        """文件名应是 1-1 这样的短名，而不是完整的环境名。"""
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        config = _play_config(tmp_path, levels=("SuperMarioBros-1-1-v0",))

        runner_mod.play(config)

        video_dir = tmp_path / "videos"
        assert (video_dir / "1-1.mp4").exists()
        assert not (video_dir / "SuperMarioBros-1-1-v0.mp4").exists()

    def test_honours_video_dir_override(self, tmp_path: Path, monkeypatch):
        custom = tmp_path / "elsewhere"
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        config = _play_config(tmp_path, levels=("SuperMarioBros-1-1-v0",), video_dir=custom)

        runner_mod.play(config)

        assert (custom / "1-1.mp4").exists(), "应使用 --video-dir 指定的目录"

    def test_forces_rgb_array_render_mode(self, tmp_path: Path, monkeypatch):
        """配置里写 human 时也必须被覆盖，否则会弹出 pygame 窗口。

        human 模式的 render() 只往 ImageViewer 推帧并返回 None，录制拿不到像素。
        """
        render_modes: list = []
        runner_mod = _patch_play(monkeypatch, _make_env_factory(render_modes=render_modes))
        config = _play_config(tmp_path, levels=("SuperMarioBros-1-1-v0",))

        runner_mod.play(config)

        assert render_modes == ["rgb_array"], "play 必须强制走 rgb_array"

    def test_keyboard_interrupt_keeps_partial_video(self, tmp_path: Path, monkeypatch):
        """Ctrl+C 中断时不该向外抛异常，且已录制的部分要保住。"""
        env = _FakeEnv(n_steps=100, interrupt_at=3)
        runner_mod = _patch_play(monkeypatch, _make_env_factory(env=env))
        config = _play_config(tmp_path, levels=("SuperMarioBros-1-1-v0",))

        runner_mod.play(config)   # 不应抛 KeyboardInterrupt

        path = tmp_path / "videos" / "1-1.mp4"
        assert path.exists(), "中断前已录的帧必须落盘"
        assert env.closed, "中断也要关掉环境"
        reader = iio.get_reader(path)
        try:
            assert reader.count_frames() >= 1
        finally:
            reader.close()

    def test_none_frame_raises_clear_error(self, tmp_path: Path, monkeypatch):
        """包装器不透传 render() 时给出可读报错，而不是 imageio 的费解堆栈。"""
        env = _FakeEnv()
        env.render = lambda: None

        runner_mod = _patch_play(monkeypatch, _make_env_factory(env=env))
        config = _play_config(tmp_path, levels=("SuperMarioBros-1-1-v0",))

        with pytest.raises(RuntimeError, match="rgb_array"):
            runner_mod.play(config)


class TestTimestampedVideoDir:
    """play 命令的输出按时间分档：每播一次新开一个目录，谁也别覆盖谁。"""

    def test_writes_into_a_timestamped_subdir(self, tmp_path: Path, monkeypatch):
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), timestamp_videos=True
        )

        runner_mod.play(config)

        video = _only_subdir(tmp_path / "videos") / "1-1.mp4"
        assert video.exists(), "视频应落在时间戳目录里"

    def test_dir_name_looks_like_a_timestamp(self, tmp_path: Path, monkeypatch):
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), timestamp_videos=True
        )

        runner_mod.play(config)

        name = _only_subdir(tmp_path / "videos").name
        # 形如 20260912_143022，或同秒重播时的 20260912_143022_2
        assert re.fullmatch(r"\d{8}_\d{6}(_\d+)?", name), f"目录名不像时间戳: {name}"

    def test_each_play_keeps_its_own_copy(self, tmp_path: Path, monkeypatch):
        """连播两场不能互相覆盖——这正是分时间目录的全部意义。"""
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), timestamp_videos=True
        )

        runner_mod.play(config)
        first = _only_subdir(tmp_path / "videos")
        runner_mod.play(config)

        dirs = sorted(p for p in (tmp_path / "videos").iterdir() if p.is_dir())
        assert len(dirs) == 2, f"两场播放应有两个目录，实际 {dirs}"
        assert (first / "1-1.mp4").exists(), "第一场的视频不该被第二场冲掉"
        assert all((d / "1-1.mp4").exists() for d in dirs)

    def test_timestamps_under_a_custom_video_dir(self, tmp_path: Path, monkeypatch):
        """--video-dir 只是换个地方放，分档行为应保持一致。"""
        custom = tmp_path / "elsewhere"
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        config = _play_config(
            tmp_path,
            levels=("SuperMarioBros-1-1-v0",),
            video_dir=custom,
            timestamp_videos=True,
        )

        runner_mod.play(config)

        assert (_only_subdir(custom) / "1-1.mp4").exists()

    def test_disabled_writes_flat(self, tmp_path: Path, monkeypatch):
        """关掉分档时按老样子平铺，训练侧录 best 模型走的就是这条。"""
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), timestamp_videos=False
        )

        runner_mod.play(config)

        assert (tmp_path / "videos" / "1-1.mp4").exists()
        assert not [p for p in (tmp_path / "videos").iterdir() if p.is_dir()], (
            "关掉分档时不该再有子目录"
        )

    def test_play_command_defaults_to_timestamped(self, tmp_path: Path, monkeypatch):
        """从真实命令行参数出发，确认默认确实是分档的。

        别的测试都直接写 SimpleNamespace，唯独这里走一遍 argparse → config，
        因为"默认开着"这件事本身只有真实参数链路能证明。
        """
        from mad_mario.config import add_play_args, config_from_play_args

        checkpoint = tmp_path / "model.zip"
        checkpoint.touch()
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        config = config_from_play_args(
            parser.parse_args(
                [
                    "--checkpoint", str(checkpoint),
                    "--save-root", str(tmp_path),
                    "--flat-save-root",     # 免去按兼容参数分子目录
                    "--levels", "1-1",
                    "--no-show",
                ]
            )
        )

        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        runner_mod.play(config)

        assert (_only_subdir(tmp_path / "videos") / "1-1.mp4").exists()


class TestSampleRuns:
    """--sample N：每关跑 N 遍，每遍各抽一个探索率采样着玩。"""

    def test_sample_run_is_named_after_its_rate(self, tmp_path: Path, monkeypatch):
        """采样播放的文件名带探索率，一眼看出这段画面有多随机。"""
        model = _FakeModel()
        runner_mod = _patch_play(monkeypatch, _make_env_factory(), model=model)
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), sample_runs=1
        )

        runner_mod.play(config)

        videos = list((tmp_path / "videos").glob("*.mp4"))
        assert len(videos) == 1
        name = videos[0].name
        assert re.fullmatch(r"1-1_\d\.\d{3}\.mp4", name), f"名字应带探索率，实际 {name}"
        # 名字里的数字必须就是这一遍实际用的值，不能各写各的
        used = {f"{rate:.3f}" for rate in model.rates_used}
        assert name.split("_", 1)[1].removesuffix(".mp4") in used

    def test_greedy_keeps_the_plain_filename(self, tmp_path: Path, monkeypatch):
        """贪心没有探索率可言，文件名维持 1-1.mp4。"""
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        config = _play_config(tmp_path, levels=("SuperMarioBros-1-1-v0",))

        runner_mod.play(config)

        assert (tmp_path / "videos" / "1-1.mp4").exists()

    def test_missing_rate_falls_back_to_the_plain_name(self):
        """没有探索率可写时用朴素名字，而不是 1-1_nan.mp4。

        非 DQN 模型（或日后换了策略）拿不到 exploration_rate，直接格式化会写出
        nan 这种谁也看不懂的文件名。
        """
        env = SimpleNamespace(world=1, level=1)

        assert _video_filename(None, env, None) == "1-1.mp4"
        assert _video_filename(None, env, float("nan")) == "1-1.mp4"
        assert _video_filename(None, env, 0.19761608882144843) == "1-1_0.198.mp4"

    def test_each_run_draws_its_own_rate(self, tmp_path: Path, monkeypatch):
        """N 遍抽 N 个不同的概率——这才是 --sample N 的意义所在。

        若改回"同一个 ε 重复 N 遍"，三遍的 rates_used 会全部相同，这里就会红。
        """
        model = _FakeModel()
        runner_mod = _patch_play(monkeypatch, _make_env_factory(), model=model)
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), sample_runs=3
        )

        runner_mod.play(config)

        assert len(set(model.rates_used)) == 3, (
            f"三遍该抽三个不同的探索率，实际用到 {set(model.rates_used)}"
        )

    def test_each_run_keeps_its_own_video(self, tmp_path: Path, monkeypatch):
        """三遍三个文件，一个都不能少——概率不同，文件名本就撞不上。"""
        model = _FakeModel()
        runner_mod = _patch_play(monkeypatch, _make_env_factory(), model=model)
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), sample_runs=3
        )

        runner_mod.play(config)

        videos = list((tmp_path / "videos").glob("*.mp4"))
        assert len(videos) == 3, f"三遍应留三个文件，实际 {[p.name for p in videos]}"

    def test_drawn_rate_stays_within_twice_the_checkpoint_rate(
        self, tmp_path: Path, monkeypatch
    ):
        """抽样范围是 [0, 2×checkpoint 的探索率]，均值仍落在原值上。

        上界不设的话，采样播放会系统性比模型本身"更随机"，看到的就不是这个
        模型的水平了。
        """
        model = _FakeModel()
        runner_mod = _patch_play(monkeypatch, _make_env_factory(), model=model)
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), sample_runs=8
        )

        runner_mod.play(config)

        assert model.rates_used, "应至少采样过一次"
        assert all(0.0 <= rate <= 2 * 0.198 for rate in model.rates_used), (
            f"抽样越界：{[r for r in model.rates_used if r > 0.396]}"
        )
        assert len(set(model.rates_used)) > 1, "每一遍都该是一个新的抽样值"

    def test_zero_rate_warns_that_nothing_varies(self, tmp_path: Path, monkeypatch, caplog):
        """探索率衰减到 0 的模型抽不出随机性，得说一声而不是静默地重播。"""
        model = _FakeModel()
        model.exploration_rate = 0.0
        runner_mod = _patch_play(monkeypatch, _make_env_factory(), model=model)
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), sample_runs=2
        )

        with caplog.at_level("WARNING"):
            runner_mod.play(config)

        assert "抽不出任何随机性" in caplog.text

    def test_missing_rate_warns_and_still_records(self, tmp_path: Path, monkeypatch, caplog):
        """拿不到探索率（非 DQN 模型）时只告警，不能把录制备份丢掉。"""
        model = _FakeModel()
        model.exploration_rate = float("nan")
        runner_mod = _patch_play(monkeypatch, _make_env_factory(), model=model)
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), sample_runs=2
        )

        with caplog.at_level("WARNING"):
            runner_mod.play(config)

        assert "抽不出任何随机性" in caplog.text
        # 文件名退化成了朴素名字，兜底要保证两遍都留得下来
        assert len(list((tmp_path / "videos").glob("*.mp4"))) == 2

    def test_each_run_resets_the_same_env(self, tmp_path: Path, monkeypatch):
        """同一关的各遍复用同一个 env：建环境很贵，reset() 本来就会重开一局。"""
        envs: list[_FakeEnv] = []
        runner_mod = _patch_play(monkeypatch, _make_env_factory(envs=envs))
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), sample_runs=3
        )

        runner_mod.play(config)

        assert len(envs) == 1, "一个关卡只该建一个环境"
        assert envs[0].resets == 3, "每遍开始都要 reset"
        assert envs[0].closed

    def test_runs_apply_to_every_level(self, tmp_path: Path, monkeypatch):
        """多关卡时，每一关都跑满 N 遍，各抽各的。"""
        model = _FakeModel()
        envs: list[_FakeEnv] = []
        runner_mod = _patch_play(monkeypatch, _make_env_factory(envs=envs), model=model)
        config = _play_config(
            tmp_path,
            levels=("SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0"),
            sample_runs=2,
        )

        runner_mod.play(config)

        videos = list((tmp_path / "videos").glob("*.mp4"))
        assert len(videos) == 4, f"两关各两遍应为 4 个文件，实际 {[p.name for p in videos]}"
        for level in ("1-1", "1-2"):
            assert len([p for p in videos if p.name.startswith(f"{level}_")]) == 2
        assert len(envs) == 2, "每个关卡各建一个环境，关内各遍复用"

    def test_every_run_samples(self, tmp_path: Path, monkeypatch):
        """N 遍都得走采样，不能只有第一遍采样、后面退回贪心。"""
        model = _FakeModel()
        runner_mod = _patch_play(monkeypatch, _make_env_factory(), model=model)
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), sample_runs=3
        )

        runner_mod.play(config)

        assert model.deterministic_calls, "应至少推理过一次"
        assert not any(model.deterministic_calls), "每一遍的每一步都该走 ε-greedy"


class TestUniquePath:
    """撞名兜底：两遍抽到同一个三位小数时，谁的录像都不能被覆盖掉。"""

    def test_keeps_a_free_name(self, tmp_path: Path):
        assert _unique_path(tmp_path, "1-1_0.198.mp4") == tmp_path / "1-1_0.198.mp4"

    def test_steps_aside_when_the_name_is_taken(self, tmp_path: Path):
        (tmp_path / "1-1_0.198.mp4").touch()

        assert _unique_path(tmp_path, "1-1_0.198.mp4").name == "1-1_0.198_2.mp4"

    def test_skips_past_several_taken_names(self, tmp_path: Path):
        for name in ("1-1_0.198.mp4", "1-1_0.198_2.mp4", "1-1_0.198_3.mp4"):
            (tmp_path / name).touch()

        assert _unique_path(tmp_path, "1-1_0.198.mp4").name == "1-1_0.198_4.mp4"


class TestSampleActions:
    """--sample：贪心策略配固定初始状态时每次播放都逐帧相同，采样才会走出新轨迹。"""

    def test_defaults_to_greedy(self, tmp_path: Path, monkeypatch):
        """默认必须还是看这个模型最好的表现，不能被采样悄悄改掉。"""
        model = _FakeModel()
        runner_mod = _patch_play(monkeypatch, _make_env_factory(), model=model)
        config = _play_config(tmp_path, levels=("SuperMarioBros-1-1-v0",))

        runner_mod.play(config)

        assert model.deterministic_calls, "应至少推理过一次"
        assert all(model.deterministic_calls), "默认应全程贪心"

    def test_sample_flag_switches_to_epsilon_greedy(self, tmp_path: Path, monkeypatch):
        """--sample 要真的改推理方式，而不只是配置里多一个字段。"""
        model = _FakeModel()
        runner_mod = _patch_play(monkeypatch, _make_env_factory(), model=model)
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), sample_runs=1
        )

        runner_mod.play(config)

        assert model.deterministic_calls, "应至少推理过一次"
        assert not any(model.deterministic_calls), "开采样后每一步都该走 ε-greedy"

    def test_logs_the_exploration_rate(self, tmp_path: Path, monkeypatch, caplog):
        """采样强度取决于 checkpoint 里的探索率，可能是 1.0；不写出来没法排查。"""
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), sample_runs=1
        )

        with caplog.at_level("INFO"):
            runner_mod.play(config)

        assert "0.198" in caplog.text, "应把作为均值的 exploration_rate 记录下来"
        assert "0.396" in caplog.text, "抽样上界（2×均值）也要写出来"
        assert "探索率=" in caplog.text, "每一遍实际用到多少，得逐遍记下来"

    def _make_load_reseeding_to_42(self, monkeypatch, runner_mod, model):
        """让假 DQN.load 复刻真实行为：按存档里的 seed 重播全局 numpy。

        存档 data 里存着 seed=42，BaseAlgorithm.__init__ 会拿它调用
        set_random_seed。这不加复刻的话，单进程内连跑两次本来就序列不同，
        这个 bug 根本测不出来——它只在"另起一个进程"时现形。
        """

        def _load(*args, **kwargs):
            np.random.seed(42)
            return model

        monkeypatch.setattr(runner_mod, "DQN", SimpleNamespace(load=_load))

    def test_each_fresh_load_still_samples_differently(self, tmp_path: Path, monkeypatch):
        """核心承诺：另跑一次命令要能看到另一种打法。

        这里每次都把种子按回 42（等价于另起一个进程加载模型），播两场必须
        走出不同的采样序列，否则 --sample 只是换了个说法的重播。
        """
        model = _FakeModel()
        runner_mod = _patch_play(monkeypatch, _make_env_factory(), model=model)
        self._make_load_reseeding_to_42(monkeypatch, runner_mod, model)
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0",), sample_runs=1
        )

        runner_mod.play(config)
        first = list(model.sampled)
        model.sampled.clear()
        runner_mod.play(config)

        assert first, "应至少采样过一次"
        assert first != model.sampled, (
            "每次加载后都该重新播种，两场播放的采样序列不该相同"
        )

    def test_greedy_leaves_the_rng_alone(self, tmp_path: Path, monkeypatch):
        """贪心不抽随机数，就不该动全局随机流。

        训练侧录最优模型是在训练进程里跑的，平白重播 numpy 会搅乱训练自身
        的随机性。
        """
        seeded: list = []
        monkeypatch.setattr(np.random, "seed", lambda *args: seeded.append(args))
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        config = _play_config(tmp_path, levels=("SuperMarioBros-1-1-v0",))

        runner_mod.play(config)

        assert seeded == [], "没开采样时不该重新播种"


class TestPlayViewer:
    """实时显示窗口的接线：窗口用假的，只验开关、生命周期与关窗语义。"""

    def _patch_viewer(
        self,
        monkeypatch,
        runner_mod,
        viewers: list[_FakeViewer],
        close_after: int | None = None,
    ):
        def _make_viewer(fps, *args, **kwargs):
            viewer = _FakeViewer(fps, close_after=close_after)
            viewers.append(viewer)
            return viewer

        monkeypatch.setattr(runner_mod, "FrameViewer", _make_viewer)
        monkeypatch.setattr(runner_mod, "is_gui_available", lambda: True)

    def test_no_show_does_not_open_window(self, tmp_path: Path, monkeypatch):
        """--no-show 时完全不该碰 matplotlib。"""
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        monkeypatch.setattr(runner_mod, "is_gui_available", lambda: True)
        monkeypatch.setattr(
            runner_mod, "FrameViewer", lambda *a, **k: pytest.fail("不该创建窗口")
        )
        config = _play_config(tmp_path, levels=("SuperMarioBros-1-1-v0",), show=False)

        runner_mod.play(config)

        assert (tmp_path / "videos" / "1-1.mp4").exists()

    def test_show_opens_one_window_and_closes_it(self, tmp_path: Path, monkeypatch):
        """整个播放过程只开一个窗口，结束时关掉，且帧确实送上屏幕。"""
        viewers: list[_FakeViewer] = []
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        self._patch_viewer(monkeypatch, runner_mod, viewers)
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0"), show=True
        )

        runner_mod.play(config)

        assert len(viewers) == 1, "两个关卡应复用同一个窗口"
        assert viewers[0].fps == 15, "节流节奏取自 --video-fps"
        assert viewers[0].shown > 0, "应有帧送上屏幕"
        assert viewers[0].closed, "播放结束必须关窗，否则进程退不掉"

    def test_falls_back_when_no_gui(self, tmp_path: Path, monkeypatch):
        """没有图形后端时降级为只录制，不能把整场播放拖垮。"""
        runner_mod = _patch_play(monkeypatch, _make_env_factory())
        monkeypatch.setattr(runner_mod, "is_gui_available", lambda: False)
        monkeypatch.setattr(
            runner_mod, "FrameViewer", lambda *a, **k: pytest.fail("无后端时不该建窗口")
        )
        config = _play_config(tmp_path, levels=("SuperMarioBros-1-1-v0",), show=True)

        runner_mod.play(config)

        assert (tmp_path / "videos" / "1-1.mp4").exists(), "降级后仍要产出视频"

    def test_closing_window_stops_playback(self, tmp_path: Path, monkeypatch):
        """关窗即停止：不继续下一关，已录制的部分保留。"""
        env = _FakeEnv(n_steps=100)
        viewers: list[_FakeViewer] = []
        runner_mod = _patch_play(monkeypatch, _make_env_factory(env=env))
        self._patch_viewer(monkeypatch, runner_mod, viewers, close_after=1)
        config = _play_config(
            tmp_path, levels=("SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0"), show=True
        )

        runner_mod.play(config)

        video_dir = tmp_path / "videos"
        assert (video_dir / "1-1.mp4").exists(), "关窗前已录的帧必须落盘"
        assert not (video_dir / "1-2.mp4").exists(), "关窗后不该再开下一关"
        assert env.closed
