"""测试 playback/viewer.py 的实时显示窗口。

用 matplotlib 的 Agg 后端跑：没有窗口可弹，但 figure、pause、关窗检测这些
逻辑与真实后端走的是同一条路径，足以覆盖节流与生命周期。
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mad_mario.playback.viewer import FrameViewer, is_gui_available  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """每个用例后清干净，避免 figure 泄漏到下一个用例影响 is_open 判断。"""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _frame(value: int = 0) -> np.ndarray:
    return np.full((8, 8, 3), fill_value=value, dtype=np.uint8)


class TestIsGuiAvailable:
    def test_returns_bool(self):
        assert isinstance(is_gui_available(), bool)

    def test_false_for_non_interactive_backend(self):
        # Agg 是非交互后端：即使建得出 figure，屏幕上也不会出现窗口，
        # 必须报 False，否则用户会以为参数没生效
        assert matplotlib.get_backend().lower() == "agg"
        assert is_gui_available() is False

    @pytest.mark.parametrize("backend", ["tkagg", "TkAgg", "qtagg"])
    def test_true_for_gui_backend(self, monkeypatch, backend: str):
        """后端名大小写不一时也要认出来。

        matplotlib 3.11 的 get_backend() 返回全小写（'tkagg'），而旧接口里的
        名字是 'TkAgg'——按字符串相等比较会误判成"没有显示器"，窗口永远不开。
        """
        monkeypatch.setattr(matplotlib, "get_backend", lambda: backend)
        assert is_gui_available() is True

    def test_false_for_unknown_backend(self, monkeypatch):
        """认不出的后端名按不可用处理，而不是抛异常拖垮整场播放。"""
        monkeypatch.setattr(matplotlib, "get_backend", lambda: "definitely-not-real")
        assert is_gui_available() is False


class TestFrameViewer:
    def test_shows_first_and_subsequent_frames(self):
        """首帧建 image、后续帧改数据，两条路径都要能跑通。"""
        viewer = FrameViewer(fps=15)
        try:
            viewer.show(_frame(0))
            viewer.show(_frame(128))
            assert viewer.is_open
        finally:
            viewer.close()

    def test_is_open_tracks_window_lifecycle(self):
        viewer = FrameViewer(fps=15)
        assert viewer.is_open, "建好后窗口是开着的"
        viewer.close()
        assert not viewer.is_open, "关掉后应报告已关闭"

    def test_low_fps_does_not_divide_by_zero(self):
        """fps 传 0 时不该崩——配置层会钳到 1，这里再兜一层。"""
        viewer = FrameViewer(fps=0)
        try:
            viewer.show(_frame())
        finally:
            viewer.close()

    def test_close_is_idempotent(self):
        """重复关闭不该报错：正常路径与异常路径都会走到 close()。"""
        viewer = FrameViewer(fps=15)
        viewer.close()
        viewer.close()
