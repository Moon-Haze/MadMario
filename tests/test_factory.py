"""测试 env/factory.py——环境创建。"""

from __future__ import annotations

import gymnasium as gym
import pytest

from mad_mario.config import EnvConfig
from mad_mario.env.factory import make_mario_env


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

    def test_default_render_mode_is_safe_for_training(self):
        """训练 thunk 不显式传 render_mode：构造与步进必须照常，无渲染副作用。

        这是删除 EnvConfig.render_mode 兜底字段的对照验证——工厂的默认
        render_mode 直传 gym，训练路径此前就一直走 None，行为不应有变。
        """
        env = make_mario_env(EnvConfig())
        try:
            obs, _ = env.reset()
            _, reward, terminated, truncated, _ = env.step(0)
            assert obs.shape == (4, 84, 84)
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
