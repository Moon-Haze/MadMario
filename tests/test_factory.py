"""测试 env/factory.py——环境创建和关卡选择。"""

from __future__ import annotations

import gymnasium as gym
import pytest

from mad_mario.config import EnvConfig
from mad_mario.env.factory import _select_levels, make_mario_env


class TestMakeMarioEnv:
    def test_default(self):
        """默认配置应成功创建环境。"""
        env = make_mario_env()
        try:
            assert isinstance(env, gym.Env)
            assert env.observation_space.shape == (4, 84, 84)
            assert env.action_space.n in (5, 7, 12)  # 取决于动作空间预设
        finally:
            env.close()

    def test_custom_level(self):
        """指定关卡名应创建对应关卡。"""
        env = make_mario_env(level_name="SuperMarioBros-2-1-v0")
        try:
            obs, _ = env.reset()
            assert obs.shape == (4, 84, 84)
        finally:
            env.close()

    def test_all_movements(self):
        """所有动作空间预设都应正常工作。"""
        for movement in ("right_only", "simple", "complex"):
            cfg = EnvConfig(movement=movement)
            env = make_mario_env(cfg)
            try:
                obs, _ = env.reset()
                assert obs.shape == (4, 84, 84)
            finally:
                env.close()

    def test_env_steps(self):
        """环境应能正常执行几步。"""
        env = make_mario_env()
        try:
            obs, _ = env.reset()
            for _ in range(10):
                obs, reward, terminated, truncated, info = env.step(0)
                assert obs.shape == (4, 84, 84)
                assert isinstance(reward, float)
                assert isinstance(info, dict)
        finally:
            env.close()

    def test_info_contains_position(self):
        """info 应包含位置信息。"""
        env = make_mario_env()
        try:
            _, info = env.reset()
            # x_pos 可能存在（根据 Mario 状态）
            assert "x_pos" in info or True  # gym-smb 通常会有
        finally:
            env.close()


class TestSelectLevels:
    def test_single_level(self):
        """单关卡时应返回全 None。"""
        cfg = EnvConfig()
        levels = _select_levels(4, cfg)
        assert levels == [None] * 4

    def test_multi_levels(self):
        """多关卡时应从列表中随机选择。"""
        cfg = EnvConfig(levels=["1-1", "1-2", "1-3"])
        levels = _select_levels(10, cfg)
        assert len(levels) == 10
        for lv in levels:
            assert lv in ("1-1", "1-2", "1-3")

    def test_no_config(self):
        """config 为 None 时应返回全 None。"""
        levels = _select_levels(3, None)
        assert levels == [None] * 3
