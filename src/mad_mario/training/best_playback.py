"""最优模型刷新时，自动把它的表现录成视频。"""

from __future__ import annotations

import logging
import math
import queue
import shutil
import threading
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

from stable_baselines3.common.callbacks import EvalCallback

from mad_mario.config import AppConfig, PlayConfig, SB3ArtifactConfig
from mad_mario.playback.runner import play

logger = logging.getLogger(__name__)

# 录制产物放在 save_root 下的这个目录里，每个最优模型一个 step_xxx 子目录
BEST_VIDEO_DIRNAME = "best_model_videos"

# 后台队列的收工信号：worker 取到它就退出，此前排下的每一趟都已录完
_STOP = object()


class _RecordingJob(NamedTuple):
    """一次待录制：哪个步数的最优模型、冻在哪个文件里、录到哪个目录。"""

    step: int
    checkpoint: Path
    video_dir: Path


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
            # 这里要展示的是这个最优模型的最好表现，贪心；采样留给命令行播放
            sample_runs=0,
        ),
    )


def _snapshot_model(best_path: Path, video_dir: Path) -> Path:
    """把刚存下的最优模型复制进这一档的视频目录，返回录制该用的 checkpoint。

    非复制不可：``best_model.zip`` 是原地覆写的同一个文件，而录制排在后台队列
    里——等轮到这条任务时它读到的可能已是更晚的那次刷新，于是目录名写着
    step_000160000，录出来却是 step_000210000 的表现。复制一份等于把模型按当时
    的样子冻住，顺带让每个视频目录自带可复现的那份权重。
    """
    video_dir.mkdir(parents=True, exist_ok=True)
    snapshot = video_dir / best_path.name
    shutil.copy2(best_path, snapshot)
    return snapshot


def _clears_the_bar(previous_best: float, current: float, min_improvement: float) -> bool:
    """这次评估够不够格算"刷新了最优模型"。

    按比例而不是固定差值：各关卡、各训练阶段的奖励量级差着数量级，写死一个
    数值要么在早期形同虚设，要么在后期永远够不着。差值乘的是上一个最优的
    绝对值，免得奖励为负时门槛反着往下滑。

    上一个最优还是 -inf（没有基准）时无条件通过——总得先有一个最优当基准。
    """
    if not math.isfinite(previous_best):
        return True
    return current >= previous_best + abs(previous_best) * min_improvement


class BestModelPlaybackCallback(EvalCallback):
    """评估回调；最优模型刷新到够格时，存档并排队录成视频。

    直接继承 EvalCallback，而不是另起一个回调去盯 best_model.zip 的改动：最优
    与否只有 EvalCallback 自己知道（best_mean_reward），从外部轮询只能靠比对
    文件时间戳，既要在每步做 stat，又会漏掉同一时间戳内的连续刷新。

    两处偏离父类的默认行为，都是为了让"最优"这两个字更有含金量：

    - **抬高门槛**：父类是评估值高一点点就刷新，于是评估噪声的每个小波峰都要
      存一次盘、录一次视频。这里要求比已采纳的最优高出一个比例才算数。
    - **录制移到后台**：一趟要跑遍全部关卡、逐步渲染加编码，就地跑完等于把
      主循环按在那儿等，评估间隔被拉长好几倍。
    """

    def __init__(
        self,
        *args,
        train_config: AppConfig,
        best_model_save_path: str | Path | None = None,
        min_improvement: float = 0.05,
        **kwargs,
    ):
        # 不把 best_model_save_path 交给父类：它的逻辑是"高一点就存"，与这里的
        # 门槛冲突。路径仍由调用方用同一个参数给出，只是存档时机改由我们掌握。
        super().__init__(*args, best_model_save_path=None, **kwargs)
        self._train_config = train_config
        self._video_root = Path(train_config.artifacts.save_root) / BEST_VIDEO_DIRNAME
        self._best_model_dir = Path(best_model_save_path) if best_model_save_path else None
        self._min_improvement = min_improvement

        self._jobs: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        # 收尾只做一次：叫停之后不再接新任务，也不重复投收工信号
        self._stopped = False
        self._lock = threading.Lock()

    def _on_step(self) -> bool:
        # 父类只在评估且成绩变好时抬高 best_mean_reward，所以前后比一次就能
        # 确定"本次评估刷新了最优"，不必自己复刻它的评估逻辑
        previous_best = self.best_mean_reward
        keep_training = super()._on_step()

        if self.best_mean_reward > previous_best:
            if _clears_the_bar(previous_best, self.best_mean_reward, self._min_improvement):
                self._adopt_best(previous_best)
            else:
                # 提升没过门槛，这次不算刷新。把标准退回去，否则它会一路漂到
                # 评估噪声的波峰上，之后真实的进步反而再也够不着
                self.best_mean_reward = previous_best

        return keep_training

    def _on_training_end(self) -> None:
        self.wait_for_recording()

    def wait_for_recording(self, timeout: float | None = None) -> None:
        """等后台把排队的视频录完再返回；幂等，正常结束与 Ctrl+C 都走这里。

        不这么等一等，进程会带着写了一半的 mp4 退出——mp4 的索引写在文件末尾，
        半截文件连播放器都打不开，同一趟里先录好的几关也跟着废掉。
        """
        with self._lock:
            worker = self._worker
            if self._stopped:
                pending = 0
            else:
                self._stopped = True
                # 先数再投哨兵，否则哨兵自己会被算成一个待录任务
                pending = self._jobs.qsize()
                if worker is not None:
                    self._jobs.put(_STOP)

        if worker is None:
            return
        if pending:
            logger.info("等待后台录制完成（还有 %d 趟未录）...", pending)
        worker.join(timeout)
        if worker.is_alive():
            logger.warning("后台录制未在超时内结束；已录好的视频保留在 %s", self._video_root)

    # ── 存档与录制 ──

    def _adopt_best(self, previous_best: float) -> None:
        """存档这个够格的最优模型，并把它排进后台录制队列。"""
        best_path = self._save_best_model()
        logger.info(
            "最优模型已刷新（步数=%d，评估奖励=%.1f%s）→ %s",
            self.num_timesteps, self.best_mean_reward,
            "" if not math.isfinite(previous_best)
            else f"，较上次 +{self.best_mean_reward - previous_best:.1f}",
            best_path if best_path is not None else "未配置存档目录",
        )
        if best_path is not None and self._train_config.play.play_on_best:
            self._record_best(best_path)

    def _save_best_model(self) -> Path | None:
        """把当前模型存成 best_model.zip；没配目录就只当评估回调使。"""
        if self._best_model_dir is None:
            return None
        self._best_model_dir.mkdir(parents=True, exist_ok=True)
        path = self._best_model_dir / "best_model.zip"
        self.model.save(str(path))
        return path

    def _record_best(self, best_path: Path) -> None:
        """把"录制这个最优模型"排进后台队列，立刻返回让训练继续。

        复制快照这一步留在主线程：它必须发生在 best_model.zip 被下一次刷新覆盖
        之前，交给后台就等于把这份文件交出去任人涂改。
        """
        # 带上步数，这样多次刷新留下的视频能按训练阶段对比着看
        video_dir = self._video_root / f"step_{self.num_timesteps:09d}"
        try:
            checkpoint = _snapshot_model(best_path, video_dir)
        except OSError as exc:
            logger.warning("复制最优模型失败（%s），跳过本次录制", exc)
            return

        if not self._enqueue(_RecordingJob(self.num_timesteps, checkpoint, video_dir)):
            logger.warning("录制队列已收工，跳过本次录制（步数=%d）", self.num_timesteps)
            return
        logger.info("已排队录制（步数=%d）→ %s", self.num_timesteps, video_dir)

    def _enqueue(self, job: _RecordingJob) -> bool:
        """把任务交给后台线程；已经叫停过就返回 False。"""
        with self._lock:
            if self._stopped:
                return False
            if self._worker is None:
                # 不是守护线程：进程退出时 Python 会先等它收尾，别让半截 mp4
                # 留在磁盘上。惰性启动，没开录制就不白养一个线程
                self._worker = threading.Thread(
                    target=self._drain_jobs, name="best-model-playback", daemon=False
                )
                self._worker.start()
            self._jobs.put(job)
            return True

    def _drain_jobs(self) -> None:
        """后台线程主体：按排队顺序一趟一趟地录，录完一个再取下一个。

        一趟失败只告警——录制是观察手段，不该拖垮训练，也不该让排在后面的
        任务跟着陪葬。
        """
        while True:
            job = self._jobs.get()
            if job is _STOP:
                return
            logger.info("后台开始录制（步数=%d）→ %s", job.step, job.video_dir)
            try:
                play(
                    _to_playback_config(self._train_config, job.checkpoint, job.video_dir),
                    # 信号只会送到主线程，这里吞不吞中断没有区别；真要停训练是
                    # 主线程的事，后台这一趟录完自己就结束了
                    catch_interrupt=False,
                )
            except Exception as exc:
                logger.warning("录制最优模型视频失败（步数=%d，%s），继续录后面的", job.step, exc)
