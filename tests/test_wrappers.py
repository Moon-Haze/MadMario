"""测试 env/wrappers.py——Mario 特有奖励包装器。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from mad_mario.env.wrappers import FlagReward, ProgressReward, ScoreReward, StuckPenalty


# ─────────────────────────────
# 辅助：创建一个基础 mock 环境
# ─────────────────────────────
class _BaseMockEnv(gym.Env):
    """可控制 info 输出的最小 mock 环境。"""

    def __init__(self):
        super().__init__()
        self.observation_space = Box(0, 255, (84, 84, 3), dtype=np.uint8)
        self.action_space = Discrete(5)
        self._step_n = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_n = 0
        return self.observation_space.sample(), self._make_info()

    def _make_info(self):
        return {}

    def step(self, action):
        self._step_n += 1
        return self.observation_space.sample(), 0.0, False, False, self._make_info()


# ─────────────────────────────
# StuckPenalty
# ─────────────────────────────
class TestStuckPenalty:
    def test_not_stuck_moves(self):
        """正常移动时不应触发卡住。"""
        class _MovingEnv(_BaseMockEnv):
            def _make_info(self):
                return {"x_pos": float(self._step_n * 10), "y_pos": 0.0}

        env = StuckPenalty(_MovingEnv(), max_stuck_steps=3, penalty=5.0)
        obs, info = env.reset()
        assert info["stuck_steps"] == 0

        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not truncated, "移动中不应被截断"
            assert info["stuck_steps"] == 0

    def test_stuck_triggers(self):
        """站在原地不动超过阈值应惩罚并截断。"""
        class _StuckEnv(_BaseMockEnv):
            def __init__(self):
                super().__init__()
                self._pos = 100.0

            def _make_info(self):
                return {"x_pos": self._pos, "y_pos": 0.0}

        env = StuckPenalty(_StuckEnv(), max_stuck_steps=3, penalty=5.0)
        env.reset()

        # 前 2 步不动 → 第 3 步触发（>= max_stuck_steps）
        for i in range(2):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not truncated, f"第 {i+1} 步不应截断"
            assert info["stuck_steps"] == i + 1

        # 第 3 步触发
        obs, reward, terminated, truncated, info = env.step(0)
        assert truncated, "应被截断"
        assert reward == -5.0, f"应获得 -5.0 惩罚，实际 {reward}"

    def test_stuck_recovers(self):
        """卡住后移动应重置计数器。"""
        class _RecoverEnv(_BaseMockEnv):
            def __init__(self):
                super().__init__()
                self._pos = 100.0

            def _make_info(self):
                return {"x_pos": self._pos, "y_pos": 0.0}

        env = StuckPenalty(_RecoverEnv(), max_stuck_steps=3, penalty=5.0)
        env.reset()

        for _ in range(2):  # 卡 2 步
            env.step(0)

        # 移动一步
        env.unwrapped._pos = 200.0
        obs, reward, terminated, truncated, info = env.step(0)
        assert not truncated, "移动后不应截断"
        assert info["stuck_steps"] == 0, "卡住计数器应重置"

    def test_stuck_tracks_max_x(self):
        """应追踪最大 x 位置。"""
        class _PosEnv(_BaseMockEnv):
            def __init__(self):
                super().__init__()
                self._pos = 0.0

            def _make_info(self):
                return {"x_pos": self._pos}

        env = StuckPenalty(_PosEnv(), max_stuck_steps=10, penalty=5.0)
        obs, info = env.reset()
        assert info["max_x_pos"] == 0.0

        env.unwrapped._pos = 50.0
        obs, reward, terminated, truncated, info = env.step(0)
        assert info["max_x_pos"] == 50.0

        env.unwrapped._pos = 30.0  # 后退
        obs, reward, terminated, truncated, info = env.step(0)
        assert info["max_x_pos"] == 50.0, "max_x_pos 不应缩小"


# ─────────────────────────────
# ProgressReward
# ─────────────────────────────
class TestProgressReward:
    def test_forward_gives_reward(self):
        """向右移动应获得额外正奖励。"""
        class _FwdEnv(_BaseMockEnv):
            def __init__(self):
                super().__init__()
                self._pos = 0.0

            def _make_info(self):
                return {"x_pos": self._pos}

        env = ProgressReward(_FwdEnv(), scale=0.1)
        env.reset()

        env.unwrapped._pos = 100.0
        obs, reward, terminated, truncated, info = env.step(0)
        # delta=100, scale=0.1 → +10
        assert reward == 10.0, f"应获得 10.0，实际 {reward}"

    def test_backward_no_penalty(self):
        """向左移动不应扣除奖励。"""
        class _FwdEnv(_BaseMockEnv):
            def __init__(self):
                super().__init__()
                self._pos = 100.0

            def _make_info(self):
                return {"x_pos": self._pos}

        env = ProgressReward(_FwdEnv(), scale=0.1)
        env.reset()

        env.unwrapped._pos = 50.0  # 后退
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == 0.0, f"后退不应扣分，实际 {reward}"

    def test_no_movement_no_reward(self):
        """静止不应获得额外奖励。"""
        class _StillEnv(_BaseMockEnv):
            def __init__(self):
                super().__init__()
                self._pos = 50.0

            def _make_info(self):
                return {"x_pos": self._pos}

        env = ProgressReward(_StillEnv(), scale=0.1)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == 0.0, f"静止不应加分，实际 {reward}"


# ─────────────────────────────
# ScoreReward
# ─────────────────────────────
class TestScoreReward:
    def test_score_increase(self):
        """得分增加应获得额外奖励。"""
        class _ScoreEnv(_BaseMockEnv):
            def __init__(self):
                super().__init__()
                self._score = 0

            def _make_info(self):
                return {"score": self._score}

        env = ScoreReward(_ScoreEnv(), scale=0.01)
        env.reset()

        env.unwrapped._score = 2000  # 踩怪 + 金币 等
        obs, reward, terminated, truncated, info = env.step(0)
        # delta=2000, scale=0.01 → +20
        assert reward == 20.0, f"应获得 20.0，实际 {reward}"

    def test_score_no_change(self):
        """得分不变不应获得额外奖励。"""
        class _NoScoreEnv(_BaseMockEnv):
            def __init__(self):
                super().__init__()
                self._score = 1000

            def _make_info(self):
                return {"score": self._score}

        env = ScoreReward(_NoScoreEnv(), scale=0.01)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == 0.0, f"得分不变不应加分，实际 {reward}"


# ─────────────────────────────
# FlagReward
# ─────────────────────────────
class TestFlagReward:
    def test_flag_get_bonus(self):
        """拉旗子时应获得额外奖励。"""
        class _FlagEnv(_BaseMockEnv):
            def _make_info(self):
                return {"flag_get": True}

        env = FlagReward(_FlagEnv(), reward_value=50.0)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == 50.0, f"应获得 50.0，实际 {reward}"

    def test_flag_not_get(self):
        """未拉旗子时不应有额外奖励。"""
        class _NoFlagEnv(_BaseMockEnv):
            def _make_info(self):
                return {"flag_get": False}

        env = FlagReward(_NoFlagEnv(), reward_value=50.0)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == 0.0, f"未拉旗子时应为 0，实际 {reward}"

    def test_flag_value_custom(self):
        """自定义奖励值应生效。"""
        class _FlagEnv(_BaseMockEnv):
            def _make_info(self):
                return {"flag_get": True}

        env = FlagReward(_FlagEnv(), reward_value=100.0)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == 100.0, f"应获得 100.0，实际 {reward}"
