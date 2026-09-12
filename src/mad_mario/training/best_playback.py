"""最优模型刷新时，自动把它的表现录成视频。"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from stable_baselines3.common.callbacks import EvalCallback

from mad_mario.config import AppConfig, PlayConfig, SB3ArtifactConfig
from mad_mario.playback.runner import play

logger = logging.getLogger(__name__)

# 录制产物放在 save_root 下的这个目录里，每个最优模型一个 step_xxx 子目录
BEST_VIDEO_DIRNAME = "best_model_videos"


def _to_playback_config(train_config: AppConfig, best_model_path: Path, video_dir: Path) -> AppConfig:
    """把训练配置改造成"只录制、不弹窗"的播放配置。

    播放的语义是观赏，不是继续训练：关掉全部奖励塑形与卡住惩罚，但保留脱困
    挣扎（否则模型卡住时画面会一直僵着）。渲染模式由 play() 自己强制成
    rgb_array，这里不用管。
    """
    env_config = replace(
        train_config.env,
        clip_rewards=False,
        stuck_penalty_enabled=False,
        stuck_rescue_enabled=True,
        progress_reward_enabled=False,
        score_reward_enabled=False,
        flag_reward_enabled=False,
    )
    return AppConfig(
        env=env_config,
        sb3=train_config.sb3,
        artifacts=SB3ArtifactConfig(checkpoint=best_model_path),
        play=PlayConfig(
            video_dir=video_dir,
            video_fps=train_config.play.video_fps,
            show=False,          # 训练途中弹窗会打断使用者，也可能根本没有显示器
            # 目录已按步数分好档，再套时间戳只会让路径变长而不多出任何信息
            timestamp_videos=False,
        ),
    )


class BestModelPlaybackCallback(EvalCallback):
    """评估回调；每当最优模型被刷新，就把它的表现录成视频。

    直接继承 EvalCallback，而不是另起一个回调去盯 best_model.zip 的改动：最优
    与否只有 EvalCallback 自己知道（best_mean_reward），从外部轮询只能靠比对
    文件时间戳，既要在每步做 stat，又会漏掉同一时间戳内的连续刷新。
    """

    def __init__(self, *args, train_config: AppConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_config = train_config
        self._video_root = Path(train_config.artifacts.save_root) / BEST_VIDEO_DIRNAME

    def _on_step(self) -> bool:
        # EvalCallback 只在评估且成绩变好时抬高 best_mean_reward，所以前后
        # 比一次就能确定"本次刚刷新了最优模型"，不必自己复刻它的判定逻辑
        previous_best = self.best_mean_reward
        keep_training = super()._on_step()

        if self.best_mean_reward > previous_best and self._train_config.play.play_on_best:
            self._record_best()

        return keep_training

    def _record_best(self) -> None:
        """录制刚保存的最优模型。失败只告警——录制是观察手段，不该拖垮训练。"""
        best_path = Path(self.best_model_save_path) / "best_model.zip"
        # 带上步数，这样多次刷新留下的视频能按训练阶段对比着看
        video_dir = self._video_root / f"step_{self.num_timesteps:09d}"
        logger.info(
            "最优模型已刷新（步数=%d，评估奖励=%.1f），开始录制 → %s",
            self.num_timesteps, self.best_mean_reward, video_dir,
        )
        try:
            play(
                _to_playback_config(self._train_config, best_path, video_dir),
                # Ctrl+C 交给训练主循环处理：这里是嵌在训练里的，吞掉中断会让
                # 使用者以为停止没生效
                catch_interrupt=False,
            )
        except Exception as exc:
            logger.warning("录制最优模型视频失败（%s），训练继续", exc)
