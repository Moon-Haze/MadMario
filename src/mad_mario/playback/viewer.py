"""把游戏帧实时显示在窗口里。

与录制器共用同一份帧：play() 取一帧，既写进视频也送来这里显示，不额外渲染。
"""

from __future__ import annotations

import time

import matplotlib

_DEFAULT_TITLE = "训练好的 Mario (SB3 DQN)"


def is_gui_available() -> bool:
    """当前环境能否真正弹出窗口。

    无显示器（服务器 / SSH）时 matplotlib 会退到 Agg 这类非交互后端，此时
    建 figure 不会报错、``plt.pause()`` 也照常返回，只是屏幕上什么都没有——
    所以要主动看后端，否则用户会以为自己参数传错了。

    ``resolve_backend()`` 把后端解析成 ``(名称, GUI 框架)``，非交互后端的
    框架是 None（Agg → ``('agg', None)``，TkAgg → ``('tkagg', 'tk')``）。
    """
    try:
        from matplotlib.backends import backend_registry

        _, gui_framework = backend_registry.resolve_backend(matplotlib.get_backend())
    except Exception:  # 认不出的后端名按"不可用"处理，不阻断播放
        return False
    return gui_framework is not None


class FrameViewer:
    """实时显示帧的 matplotlib 窗口。

    负责按视频帧率节流：显示节奏与录制出的视频回放速度一致，所见即所得。
    """

    def __init__(self, fps: int, title: str = _DEFAULT_TITLE):
        import matplotlib.pyplot as plt

        self._plt = plt
        self._interval = 1.0 / max(1, fps)
        self._last: float | None = None

        plt.ion()
        self._fig, self._ax = plt.subplots()
        self._ax.axis("off")
        try:
            self._fig.canvas.manager.set_window_title(title)  # type: ignore[union-attr]
        except AttributeError:
            # 内嵌 canvas 之类的后端没有窗口管理器，标题设不上不影响显示
            pass
        self._image = None

    def show(self, frame) -> None:
        """显示一帧，并让 GUI 事件循环推进到下一帧的时刻。"""
        if self._image is None:
            self._image = self._ax.imshow(frame)
        else:
            self._image.set_data(frame)
        self._fig.canvas.draw_idle()

        now = time.monotonic()
        if self._last is None:
            delay = 0.0
        else:
            delay = max(0.0, self._interval - (now - self._last))

        if delay > 0:
            self._plt.pause(delay)
        else:
            # 首帧以及"单步耗时已超过帧间隔"时 delay 为 0，此时绝不能调
            # pause()：matplotlib 把 timeout<=0 解释成"无限等待"，会直接挂死。
            # 推进一次事件循环就够了，节流本来也不需要等。
            self._fig.canvas.flush_events()

        # 基准取"这一帧真正显示完"的时刻，而不是进入 show() 的时刻——否则
        # 下一帧会把本帧的等待时间又算进耗时里重复扣除，节奏越跑越快
        self._last = time.monotonic()

    @property
    def is_open(self) -> bool:
        """窗口是否还在。用户关掉窗口即表示停止播放。"""
        return bool(self._plt.get_fignums())

    def close(self) -> None:
        self._plt.ioff()
        self._plt.close(self._fig)
