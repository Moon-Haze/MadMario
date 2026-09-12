"""把游戏画面帧写入视频文件。"""

from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np

# 视频编码质量（imageio 的 0-10 档，越大越清晰、文件也越大）
VIDEO_QUALITY = 7


class VideoRecorder:
    """把一个关卡的游戏帧写成单个 mp4 文件。

    用上下文管理器保证异常路径下 writer 也被关闭：关卡中途报错或被 Ctrl+C
    打断时，既不能泄漏 ffmpeg 子进程句柄，也不能留下没写完的坏文件。
    """

    def __init__(self, path: Path, fps: int):
        self.path = path
        self.fps = fps
        self._writer = None
        self._frames = 0

    @property
    def frame_count(self) -> int:
        """已写入的帧数，用于在日志里汇报录制结果。"""
        return self._frames

    def __enter__(self) -> VideoRecorder:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            self.path,
            fps=self.fps,
            codec="libx264",
            quality=VIDEO_QUALITY,
            # 帧尺寸未必是 16 的倍数；交给 ffmpeg 处理，避免被隐式缩放
            macro_block_size=None,
        )
        return self

    def append(self, frame: np.ndarray) -> None:
        """追加一帧（H×W×3 的 uint8 数组）。"""
        self._writer.append_data(frame)
        self._frames += 1

    def __exit__(self, *exc_info) -> bool:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return False  # 不吞异常，让调用方照常看到失败原因
