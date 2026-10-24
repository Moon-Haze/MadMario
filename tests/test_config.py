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
    _levels_type,
    _parse_world_token,
    _worlds_type,
    add_play_args,
    add_train_args,
    build_env_name,
    build_parser,
    config_from_play_args,
    config_from_train_args,
    playback_env_config,
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

    def test_n_steps_and_exploration_steps(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(["--n-steps", "3", "--exploration-steps", "5000000"])
        config = config_from_train_args(args)
        assert config.sb3.n_steps == 3
        assert config.sb3.exploration_steps == 5_000_000

    def test_exploration_steps_defaults_to_none(self):
        """None 表示沿用 --exploration-fraction 的比例语义。"""
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        config = config_from_train_args(parser.parse_args([]))
        assert config.sb3.exploration_steps is None

    def test_eval_levels(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        config = config_from_train_args(parser.parse_args(["--eval-levels", "6"]))
        assert config.artifacts.eval_levels == 6

    def test_eval_levels_floors_at_one(self):
        """0 会让评估环境数变成 0，直接起不来。"""
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        config = config_from_train_args(parser.parse_args(["--eval-levels", "0"]))
        assert config.artifacts.eval_levels == 1

    def test_save_freq(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        config = config_from_train_args(parser.parse_args(["--save-freq", "250000"]))
        assert config.artifacts.save_freq == 250_000

    def test_reward_clip_defaults_off(self):
        """裁剪会把 −25 的死亡惩罚压成 −1，默认不该开。"""
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        config = config_from_train_args(parser.parse_args([]))
        assert config.env.clip_rewards is False

    def test_reward_clip_can_be_turned_on(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        config = config_from_train_args(parser.parse_args(["--reward-clip"]))
        assert config.env.clip_rewards is True

    def test_no_reward_clip_still_accepted(self):
        """旧命令行的 --no-reward-clip 必须继续可用，否则既有的脚本会解析失败。"""
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        config = config_from_train_args(parser.parse_args(["--no-reward-clip"]))
        assert config.env.clip_rewards is False

    def test_bigger_defaults(self):
        """缓冲区是总转移数，会被除以并行环境数；5 万配 24 个环境留不住成功样本。"""
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        config = config_from_train_args(parser.parse_args([]))
        assert config.sb3.buffer_size == 100_000
        assert config.sb3.batch_size == 64


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


# ─────────────────────────────
# playback_env_config
# ─────────────────────────────
class TestPlaybackEnvConfig:
    """播放语义单点：六个开关的取值与其余字段的保留。"""

    def test_playback_switches(self):
        """训练语义全开的配置变换后应只留脱困。"""
        train_cfg = EnvConfig(clip_rewards=True, stuck_penalty_enabled=True,
                              progress_reward_enabled=True, score_reward_enabled=True,
                              flag_reward_enabled=True, stuck_rescue_enabled=False)
        play_cfg = playback_env_config(train_cfg)
        assert play_cfg.clip_rewards is False
        assert play_cfg.stuck_penalty_enabled is False
        assert play_cfg.progress_reward_enabled is False
        assert play_cfg.score_reward_enabled is False
        assert play_cfg.flag_reward_enabled is False
        assert play_cfg.stuck_rescue_enabled is True

    def test_other_fields_preserved(self):
        """关卡、动作预设、观测参数等字段应原样保留。"""
        train_cfg = EnvConfig(game="SuperMarioBros", world=2, level=3, version=1,
                              levels=["SuperMarioBros-1-2-v0"], movement="complex",
                              frame_skip=6, frame_stack=3, resize_shape=64,
                              stuck_max_steps=42, flag_reward_value=99.0)
        play_cfg = playback_env_config(train_cfg)
        assert play_cfg.game == "SuperMarioBros"
        assert (play_cfg.world, play_cfg.level, play_cfg.version) == (2, 3, 1)
        assert play_cfg.levels == ["SuperMarioBros-1-2-v0"]
        assert play_cfg.movement == "complex"
        assert play_cfg.frame_skip == 6
        assert play_cfg.frame_stack == 3
        assert play_cfg.resize_shape == 64
        # 与塑形无关的卡住参数也保留——脱困还要用它们
        assert play_cfg.stuck_max_steps == 42

    def test_input_not_mutated(self):
        """变换是纯函数：原配置不被改写。"""
        train_cfg = EnvConfig(clip_rewards=True)
        playback_env_config(train_cfg)
        assert train_cfg.clip_rewards is True
