"""测试 config.py——CLI 参数解析、配置构建等纯函数。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from mad_mario.config import (
    MAX_LEVEL,
    MAX_WORLD,
    SB3ArtifactConfig,
    _expand_level,
    _format_worlds,
    _levels_type,
    _parse_world_token,
    _worlds_type,
    add_play_args,
    add_train_args,
    build_env_name,
    build_parser,
    compatible_save_root,
    config_from_play_args,
    config_from_train_args,
    resolve_checkpoint,
)
from mad_mario.config import EnvConfig


# ─────────────────────────────
# _parse_world_token
# ─────────────────────────────
class TestParseWorldToken:
    def test_single(self):
        assert _parse_world_token("3") == [3]

    def test_range(self):
        assert _parse_world_token("1-3") == [1, 2, 3]

    def test_range_reversed(self):
        with pytest.raises(argparse.ArgumentTypeError, match="起始值"):
            _parse_world_token("5-3")

    def test_out_of_bounds_low(self):
        with pytest.raises(argparse.ArgumentTypeError, match="超出范围"):
            _parse_world_token("0")

    def test_out_of_bounds_high(self):
        with pytest.raises(argparse.ArgumentTypeError, match="超出范围"):
            _parse_world_token(str(MAX_WORLD + 1))

    def test_non_numeric(self):
        with pytest.raises(argparse.ArgumentTypeError, match="格式错误"):
            _parse_world_token("abc")


# ─────────────────────────────
# _worlds_type
# ─────────────────────────────
class TestWorldsType:
    def test_single(self):
        assert _worlds_type("1") == [1]

    def test_commas(self):
        assert _worlds_type("1,3,5") == [1, 3, 5]

    def test_range(self):
        assert _worlds_type("1-3") == [1, 2, 3]

    def test_mixed(self):
        assert _worlds_type("1-3,5,7-8") == [1, 2, 3, 5, 7, 8]
        # 验证去重
        assert _worlds_type("1-3,2-4") == [1, 2, 3, 4]

    def test_empty(self):
        with pytest.raises(argparse.ArgumentTypeError, match="不能为空"):
            _worlds_type("")

    def test_all_worlds(self):
        result = _worlds_type(f"1-{MAX_WORLD}")
        assert len(result) == MAX_WORLD
        assert result == list(range(1, MAX_WORLD + 1))


# ─────────────────────────────
# _levels_type
# ─────────────────────────────
class TestLevelsType:
    def test_single_number(self):
        assert _levels_type("1") == ["1"]

    def test_world_level(self):
        assert _levels_type("1-1,2-1") == ["1-1", "2-1"]

    def test_multiple(self):
        assert _levels_type("1-1,1-2,1-3") == ["1-1", "1-2", "1-3"]

    def test_empty(self):
        with pytest.raises(argparse.ArgumentTypeError, match="不能为空"):
            _levels_type("")

    def test_invalid_format(self):
        with pytest.raises(argparse.ArgumentTypeError, match="必须为数字"):
            _levels_type("1-")

    def test_out_of_bounds_world(self):
        with pytest.raises(argparse.ArgumentTypeError, match="世界号"):
            _levels_type("9-1")

    def test_out_of_bounds_level(self):
        with pytest.raises(argparse.ArgumentTypeError, match="关卡号"):
            _levels_type(f"1-{MAX_LEVEL + 1}")


# ─────────────────────────────
# 工具函数
# ─────────────────────────────
class TestBuildEnvName:
    def test_default(self):
        assert build_env_name("SuperMarioBros", 1, 1) == "SuperMarioBros-1-1-v0"

    def test_version(self):
        assert build_env_name("SuperMarioBros", 2, 3, 1) == "SuperMarioBros-2-3-v1"


class TestExpandLevel:
    def test_short(self):
        assert _expand_level("1-1") == "SuperMarioBros-1-1-v0"

    def test_already_full(self):
        assert _expand_level("SuperMarioBros-2-1-v0") == "SuperMarioBros-2-1-v0"


class TestFormatWorlds:
    def test_single(self):
        assert _format_worlds([5]) == "5"

    def test_consecutive(self):
        assert _format_worlds([1, 2, 3, 4]) == "1-4"

    def test_non_consecutive(self):
        assert _format_worlds([1, 3, 5]) == "1,3,5"


# ─────────────────────────────
# compatible_save_root
# ─────────────────────────────
class TestCompatibleSaveRoot:
    def test_single_world_level(self):
        cfg = EnvConfig(world=1, level=2, movement="right_only")
        result = compatible_save_root(Path("ckpt"), cfg)
        assert result == Path("ckpt/1-2-v0_right_only")

    def test_multi_worlds(self):
        cfg = EnvConfig(worlds=[1, 2, 3], movement="simple", version=1)
        result = compatible_save_root(Path("ckpt"), cfg)
        assert "worlds-1-3" in str(result)

    def test_multi_levels(self):
        cfg = EnvConfig(levels=["1-1", "1-2", "1-3"], movement="complex")
        result = compatible_save_root(Path("ckpt"), cfg)
        assert "multi" in str(result)


# ─────────────────────────────
# resolve_checkpoint
# ─────────────────────────────
class TestResolveCheckpoint:
    def test_no_resume(self):
        cfg = SB3ArtifactConfig(resume=False)
        assert resolve_checkpoint(cfg) is None

    def test_explicit_checkpoint(self):
        path = Path("/some/path.zip")
        cfg = SB3ArtifactConfig(checkpoint=path, resume=True)
        assert resolve_checkpoint(cfg) == path

    def test_nonexistent(self, tmp_path: Path):
        cfg = SB3ArtifactConfig(save_root=tmp_path, resume=True)
        assert resolve_checkpoint(cfg) is None

    def test_final_exists(self, tmp_path: Path):
        final = tmp_path / "mario_dqn_final.zip"
        final.touch()
        cfg = SB3ArtifactConfig(save_root=tmp_path, resume=True)
        assert resolve_checkpoint(cfg) == final

    def test_snapshots_subdir_discovered(self, tmp_path: Path):
        # 周期档存在 save_root/snapshots/ 子目录时也应被自动发现（此前漏扫会返回 None）
        snap = tmp_path / "snapshots"
        snap.mkdir()
        ckpt = snap / "mario_dqn_199968_steps.zip"
        ckpt.touch()
        cfg = SB3ArtifactConfig(save_root=tmp_path, resume=True)
        assert resolve_checkpoint(cfg) == ckpt

    def test_steps_chosen_over_final(self, tmp_path: Path):
        # 只要存在周期档就优先于 final.zip；final.zip 仅在无周期档时兜底
        snap = tmp_path / "snapshots"
        snap.mkdir()
        ckpt = snap / "mario_dqn_299952_steps.zip"
        ckpt.touch()
        final = tmp_path / "mario_dqn_final.zip"
        final.touch()
        cfg = SB3ArtifactConfig(save_root=tmp_path, resume=True)
        assert resolve_checkpoint(cfg) == ckpt

    def test_largest_steps_wins_across_dirs(self, tmp_path: Path):
        # 顶层与 snapshots/ 并存时按训练步数取最大，而非按修改时间取最新
        old_big = tmp_path / "mario_dqn_399936_steps.zip"
        old_big.touch()
        os.utime(old_big, (946684800, 946684800))  # mtime 设为 2000-01-01，故意更旧
        snap = tmp_path / "snapshots"
        snap.mkdir()
        new_small = snap / "mario_dqn_299952_steps.zip"
        new_small.touch()
        cfg = SB3ArtifactConfig(save_root=tmp_path, resume=True)
        assert resolve_checkpoint(cfg) == old_big


# ─────────────────────────────
# config_from_train_args
# ─────────────────────────────
class TestConfigFromTrainArgs:
    def test_minimal(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(
            [
                "--timesteps",
                "1000",
            ]
        )
        config = config_from_train_args(args)
        assert config.sb3.total_timesteps == 1000
        assert config.env.world == 1
        assert config.env.level == 1

    def test_with_worlds(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(
            [
                "--worlds",
                "1-3",
            ]
        )
        config = config_from_train_args(args)
        assert config.env.worlds == [1, 2, 3]
        assert "1-1" in config.env.levels[0]
        assert len(config.env.levels) == 12  # 3 worlds × 4 levels

    def test_with_levels(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(
            [
                "--levels",
                "1-1,2-1",
                "--movement",
                "simple",
            ]
        )
        config = config_from_train_args(args)
        assert len(config.env.levels) == 2
        assert "1-1" in config.env.levels[0]

    def test_reward_flags(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(
            [
                "--no-progress-reward",
                "--no-score-reward",
                "--no-flag-reward",
            ]
        )
        config = config_from_train_args(args)
        assert config.env.progress_reward_enabled is False
        assert config.env.score_reward_enabled is False
        assert config.env.flag_reward_enabled is False

    def test_sb3_params(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(
            [
                "--learning-rate",
                "0.001",
                "--batch-size",
                "64",
                "--gamma",
                "0.95",
            ]
        )
        config = config_from_train_args(args)
        assert config.sb3.learning_rate == 0.001
        assert config.sb3.batch_size == 64
        assert config.sb3.gamma == 0.95

    def test_log_level_default(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args([])
        config = config_from_train_args(args)
        assert config.artifacts.log_level == "info"

    def test_log_level_custom(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(["--log-level", "debug"])
        config = config_from_train_args(args)
        assert config.artifacts.log_level == "debug"


# ─────────────────────────────
# config_from_play_args
# ─────────────────────────────
class TestConfigFromPlayArgs:
    def test_minimal(self):
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args([])
        config = config_from_play_args(args)
        # 播放模式关闭所有奖励
        assert config.env.clip_rewards is False
        assert config.env.stuck_penalty_enabled is False
        assert config.env.progress_reward_enabled is False
        assert config.env.score_reward_enabled is False
        assert config.env.flag_reward_enabled is False

    def test_custom_level(self):
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args(["--levels", "1-1,1-2"])
        config = config_from_play_args(args)
        assert len(config.env.levels) == 2


# ─────────────────────────────
# build_parser
# ─────────────────────────────
class TestBuildParser:
    def test_has_subcommands(self):
        parser = build_parser()
        assert "train" in parser._subparsers._group_actions[0].choices  # type: ignore[union-attr]
        assert "play" in parser._subparsers._group_actions[0].choices  # type: ignore[union-attr]

    def test_train_help(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["train", "--help"])
        assert exc.value.code == 0

    def test_play_help(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["play", "--help"])
        assert exc.value.code == 0


# ─────────────────────────────
# 卡住脱困配置
# ─────────────────────────────
class TestStuckRescueConfig:
    def test_env_config_defaults(self):
        cfg = EnvConfig()
        assert cfg.stuck_progress_epsilon == 8.0
        assert cfg.stuck_rescue_enabled is True
        assert cfg.stuck_rescue_max_steps == 120

    def test_cli_flags_map_to_env_config(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(
            [
                "--no-stuck-rescue",
                "--stuck-rescue-max-steps",
                "7",
                "--stuck-progress-epsilon",
                "3.5",
            ]
        )
        config = config_from_train_args(args)
        assert config.env.stuck_rescue_enabled is False
        assert config.env.stuck_rescue_max_steps == 7
        assert config.env.stuck_progress_epsilon == 3.5

    def test_rescue_max_steps_clamped_to_one(self):
        # 与 stuck_max_steps 一致：下限钳制为 1，避免预算为 0 时每步都判负
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(["--stuck-rescue-max-steps", "0"])
        config = config_from_train_args(args)
        assert config.env.stuck_rescue_max_steps == 1

    def test_play_enables_rescue_without_penalty(self):
        # 播放语义：不扣分、不因卡住提前结束关卡，但必须随机挣扎脱困
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args([])
        config = config_from_play_args(args)
        assert config.env.stuck_penalty_enabled is False
        assert config.env.stuck_rescue_enabled is True


# ─────────────────────────────
# 播放录制配置
# ─────────────────────────────
class TestPlayConfig:
    def test_cli_flags_map_to_play_config(self):
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args(["--video-fps", "30", "--video-dir", "/tmp/v"])
        config = config_from_play_args(args)
        assert config.play.video_fps == 30
        assert config.play.video_dir == Path("/tmp/v")

    def test_defaults(self):
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args([])
        config = config_from_play_args(args)
        # 15 = 60 / frame_skip(4)，即还原真实播放速度的帧率
        assert config.play.video_fps == 15
        assert config.play.video_dir is None   # None 表示用 save_root/videos

    def test_fps_clamped_to_one(self):
        # 0 或负帧率会让编码器直接失败，下限钳制为 1
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args(["--video-fps", "0"])
        config = config_from_play_args(args)
        assert config.play.video_fps == 1

    def test_train_config_keeps_default_play(self):
        # AppConfig 新增的 play 字段带默认值，训练路径不传也不该受影响
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args([])
        config = config_from_train_args(args)
        assert config.play.video_fps == 15

    def test_show_defaults_to_on(self):
        # 默认边显示边录制；无 GUI 环境由 play() 自行降级，而非靠这里改默认值
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args([])
        config = config_from_play_args(args)
        assert config.play.show is True

    def test_no_show_flag(self):
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args(["--no-show"])
        config = config_from_play_args(args)
        assert config.play.show is False

    def test_timestamp_videos_defaults_to_on(self):
        # 命令行播放默认按时间分档，免得反复播放互相覆盖
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args([])
        config = config_from_play_args(args)
        assert config.play.timestamp_videos is True

    def test_play_on_best_defaults_to_on(self):
        # 训练中每次刷新最优模型都录一段，是 --no-play-best 要关掉的行为
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args([])
        config = config_from_train_args(args)
        assert config.play.play_on_best is True

    def test_sample_runs_defaults_to_zero(self):
        # 0 = 不采样、只贪心播一遍。不该因为新增开关就把"看模型最好表现"改掉
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args([])
        config = config_from_play_args(args)
        assert config.play.sample_runs == 0

    def test_bare_sample_flag_means_one_run(self):
        # 裸写 --sample 等价于 --sample 1，老用法不至于失效
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args(["--sample"])
        config = config_from_play_args(args)
        assert config.play.sample_runs == 1

    def test_sample_takes_a_number(self):
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args(["--sample", "3"])
        config = config_from_play_args(args)
        assert config.play.sample_runs == 3

    def test_sample_clamped_to_one(self):
        # "一遍都不采样"没有意义，下限钳制为 1（与 --video-fps 的钳制同理）
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args(["--sample", "0"])
        config = config_from_play_args(args)
        assert config.play.sample_runs == 1

    def test_no_play_best_flag(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(["--no-play-best"])
        config = config_from_train_args(args)
        assert config.play.play_on_best is False
