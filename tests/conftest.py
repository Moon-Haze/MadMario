"""共享 fixture 和 mock 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete


class MockMarioEnv(gym.Env):
    """模拟的 Mario 环境，用于包装器单元测试。

    每次 ``step()`` 根据预设的 info 序列返回数据，让测试可以精确控制
    x_pos / score / flag_get 等字段。
    """

    def __init__(self, info_sequence: list[dict[str, Any]] | None = None):
        super().__init__()
        self.observation_space = Box(0, 255, (84, 84, 3), dtype=np.uint8)
        self.action_space = Discrete(5)
        self._info_seq = info_sequence or [{}]
        self._step_count = 0
        self._reset_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._reset_count += 1
        info = self._info_seq[0].copy() if self._info_seq else {}
        return self.observation_space.sample(), info

    def step(self, action):
        idx = min(self._step_count, len(self._info_seq) - 1)
        info = self._info_seq[idx].copy()
        self._step_count += 1
        return (
            self.observation_space.sample(),  # obs
            0.0,                               # reward
            False,                             # terminated
            False,                             # truncated
            info,
        )


@pytest.fixture
def mock_env():
    """返回一个无 info 的 mock 环境，包装器测试中可直接替换。"""
    return MockMarioEnv()


@pytest.fixture
def tmp_save_dir(tmp_path: Path) -> Path:
    """临时保存目录，用于测试 CSV / 图表输出。"""
    return tmp_path / "saves"
