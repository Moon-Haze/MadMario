"""测试 artifacts.py——checkpoint 发现与目录命名的磁盘策略。"""

from __future__ import annotations

import os
from pathlib import Path

from mad_mario.artifacts import (
    _format_worlds,
    compatible_save_root,
    resolve_checkpoint,
)
from mad_mario.config import EnvConfig, SB3ArtifactConfig


class TestFormatWorlds:
    def test_single(self):
        assert _format_worlds([5]) == "5"

    def test_consecutive(self):
        assert _format_worlds([1, 2, 3, 4]) == "1-4"

    def test_non_consecutive(self):
        assert _format_worlds([1, 3, 5]) == "1,3,5"


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
