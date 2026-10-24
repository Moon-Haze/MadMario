"""加载一个 checkpoint，跑完各关卡并把画面录成视频。

放在 playback 下而不是 cli 里，是为了让训练侧（best 模型刷新时自动录一段）
也能复用：trainer → playback.runner 是单向依赖，不会和 cli → trainer 成环。

编排住在 PlaybackSession 里：模型加载、环境工厂、显示窗口、线程额度都是
构造参数，测试在接缝上传假件，不必再 patch 模块全局。play() 只是命令行的
薄入口——比 session 多做的一件事是消化 Ctrl+C。
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
from stable_baselines3 import DQN

from mad_mario.config import AppConfig
from mad_mario.env.factory import make_mario_env
from mad_mario.playback.video import VideoRecorder
from mad_mario.playback.viewer import FrameViewer, is_gui_available

logger = logging.getLogger(__name__)

# 录制只认 rgb_array。human 模式会弹出 pygame 窗口并让 render() 返回 None，
# 与"播放不弹窗"的目标相悖——所以这里覆盖配置里的 render_mode，而不是照搬。
RECORD_RENDER_MODE = "rgb_array"


class _Playthrough(NamedTuple):
    """一趟独立的播放：一个探索率、一个关卡、一个输出文件。"""

    level_name: str | None
    level_label: str
    run_idx: int
    runs: int
    # 这一趟实际要用的探索率；贪心播放没有，为 None
    rate: float | None
    video_path: Path
    # 相对 video_dir 的路径，进日志用——绝对路径太长，看的人只关心目录树
    shown_path: str

    @property
    def progress(self) -> str:
        return f"（第 {self.run_idx}/{self.runs} 遍）" if self.runs > 1 else ""


class _LevelPlan(NamedTuple):
    """一个关卡要录的全部趟次，顺序即播放顺序。"""

    level_name: str | None
    label: str
    playthroughs: list[_Playthrough]


def _level_short_name(level_name, config):
    """把环境名 SuperMarioBros-1-1-v0 压成 1-1，用作视频文件名。"""
    if level_name:
        match = re.search(r"-(\d+)-(\d+)-v\d+$", level_name)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return level_name
    return f"{config.world}-{config.level}"


def _render_frame(env):
    """取当前画面帧。

    模式已由会话强制为 RECORD_RENDER_MODE，正常不会拿到空帧；这里兜底
    是为了在包装器不透传 render() 时给出可读的报错，而不是让 imageio 在
    内部抛出难懂的堆栈。
    """
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            f"环境未返回画面帧，录制需要 {RECORD_RENDER_MODE} 渲染模式"
        )
    return frame


def _emit_frame(env, recorder, viewer):
    """取一帧画面，写进视频，并（可选）送上实时窗口。

    同一份帧两用：录制与显示共用一次 render()，不额外渲染。
    """
    frame = _render_frame(env)
    recorder.append(frame)
    if viewer is not None:
        viewer.show(frame)


def _open_viewer(play_config):
    """按配置打开实时显示窗口；开不了就降级为只录制。

    没有显示器时 matplotlib 会退到非交互后端，此时建 figure 不报错、屏幕上
    却什么都没有——所以主动检测，把"看不见画面"变成一条明确的警告，而不是
    让用户对着一个不存在的窗口纳闷。
    """
    if not play_config.show:
        return None
    if not is_gui_available():
        logger.warning("当前环境没有可用的图形后端，跳过实时显示，只录制视频")
        return None
    try:
        return FrameViewer(play_config.video_fps)
    except Exception as exc:
        logger.warning("打开显示窗口失败（%s），跳过实时显示，只录制视频", exc)
        return None


def _run_episode(env, model, recorder, state, viewer=None, deterministic=True):
    """跑完一个回合，全程把画面写进录制器。

    返回 (总奖励, 步数, 是否通关, 是否因关窗而中断)。
    """
    _emit_frame(env, recorder, viewer)

    total_reward = 0.0
    step = 0
    flag_get = False

    while True:
        # 关窗检查放在取帧之前：窗口已经没了还往上画，matplotlib 会报错或
        # 悄悄重建一个，两者都不是用户想要的
        if viewer is not None and not viewer.is_open:
            return total_reward, step, False, True

        action, _ = model.predict(state, deterministic=deterministic)
        state, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        step += 1

        # 先录帧再判结束，这样终局画面也在视频里
        _emit_frame(env, recorder, viewer)

        if terminated or truncated or info.get("flag_get"):
            flag_get = bool(info.get("flag_get", False))
            break

    return total_reward, step, flag_get, False


def _video_filename(level_name, env_config) -> str:
    """关卡视频文件名：一律是干净的 1-1.mp4。

    随机强度不进文件名，交给所在目录承担（见 _rate_dirname）——文件名留给关卡，
    目录名留给这一遍的随机强度，两者各司其职；也不必为了塞进第二个信息让文件名
    越接越长。
    """
    return f"{_level_short_name(level_name, env_config)}.mp4"


def _rate_dirname(exploration_rate):
    """这一遍的探索率对应的子目录名（0.041 → 0-041）；贪心播放没有，返回 None。

    点号换成短横是为了当目录名用：目录名里带点容易和扩展名混淆，在 shell 补全
    和 glob 里也碍事，横杠干净得多。
    """
    if exploration_rate is None or not math.isfinite(exploration_rate):
        return None
    return f"{exploration_rate:.3f}".replace(".", "-")


def _rate_label(exploration_rate) -> str:
    """日志里的探索率写法；贪心播放没有探索率可说，写"贪心"而不是 nan。"""
    if exploration_rate is None or not math.isfinite(exploration_rate):
        return "贪心"
    return f"{exploration_rate:.3f}"


def _unique_path(directory: Path, filename: str, taken: set[Path] | None = None) -> Path:
    """给文件名让路，避免同一目录下的两个视频互相覆盖。

    每遍的探索率是各自抽的，概率不同就落在不同目录、正常撞不上；但目录名只
    保留三位小数，两遍抽到同一个值的概率并非为零，而覆盖掉一整段录像连个痕迹
    都不会留下。撞名时加 _2、_3 往后顺延——这是意外情况的兜底，正常路径上
    文件名保持干净。

    taken 是本轮已经占下的路径集合，选中的名字会加进去。并发播放时靠它把
    "看一眼没人用就占上"这一步挪到派发之前，否则两个线程同时问同一个名字都会
    得到"没人用"，兜底就形同虚设。
    """
    taken = set() if taken is None else taken
    stem, suffix = Path(filename).stem, Path(filename).suffix
    path = directory / filename
    idx = 2
    while path.exists() or path in taken:
        path = directory / f"{stem}_{idx}{suffix}"
        idx += 1
    taken.add(path)
    return path


def _has_exploration(base_rate) -> bool:
    """这个探索率还有没有随机性可抽。

    缺失（非 DQN 模型）与非正（已衰减到 0）都意味着抽多少次都是 0，分不出来，
    调用方据此跳过取样、也别去改写模型。
    """
    return base_rate is not None and math.isfinite(base_rate) and base_rate > 0


def _draw_exploration_rate(base_rate):
    """给这一遍抽一个探索率：以 checkpoint 的值为均值，在 [0, 2×均值] 上均匀取。

    范围对称地跨在均值两侧，均值因此还是 checkpoint 原本的强度——玩得多了整体
    水平仍是这个模型的水准，不会一开采样就系统性地偏随机或偏贪心。

    抽不出来时返回 None，调用方据此保留 checkpoint 原值。
    """
    if not _has_exploration(base_rate):
        return None
    return float(np.random.uniform(0.0, 2.0 * base_rate))


def _reseed_for_sampling() -> None:
    """把随机性交还给操作系统熵，让每次播放的采样序列都不同。

    DQN.load 会用存档里的 seed（本项目是 42）重新播种 numpy，于是每个进程的
    ε-greedy 序列逐位相同——"再跑一次换种打法"就成了空话。实测两次独立运行
    开 --sample 产出的 mp4 md5 一致，正是这个原因。
    """
    np.random.seed(None)


def _playback_threads() -> int:
    """无窗口播放开几个线程：CPU 核心数的一半，至少 1。

    渲染、推理、编码全是 CPU 活，按核心数开满只会让它们互相抢；留一半给系统
    和其他进程，整体反而更快。
    """
    return max(1, (os.cpu_count() or 2) // 2)


def _plan_playthroughs(video_dir, levels, env_config, runs, rates) -> list[_LevelPlan]:
    """把要录的每一趟排好，并把输出路径在这一步就定死。

    路径必须在派发线程之前定好（见 _unique_path 的 taken）。计划本身与执行
    方式无关：串行还是并发，录出来的目录树完全一样。
    """
    taken: set[Path] = set()
    plan = []
    for level_name in levels:
        label = level_name or f"世界{env_config.world}-{env_config.level}"
        playthroughs = []
        for run_idx in range(1, runs + 1):
            rate = rates[run_idx - 1]
            # 探索率决定子目录（0-041/），关卡决定文件名（1-1.mp4）；贪心播放
            # 没有探索率，直接平铺在 video_dir 下
            dirname = _rate_dirname(rate)
            run_dir = video_dir / dirname if dirname else video_dir
            path = _unique_path(run_dir, _video_filename(level_name, env_config), taken)
            playthroughs.append(_Playthrough(
                level_name=level_name,
                level_label=label,
                run_idx=run_idx,
                runs=runs,
                rate=rate,
                video_path=path,
                shown_path=str(path.relative_to(video_dir)),
            ))
        plan.append(_LevelPlan(level_name=level_name, label=label,
                               playthroughs=playthroughs))
    return plan


def _record_playthrough(playthrough, env, model, config, deterministic, viewer=None) -> bool:
    """跑完一趟并把画面录下来。返回是否因为显示窗口被关掉而中断。

    串行与并发两条路都走这里，免得录制接线出现两份、改一处漏一处。
    """
    if not deterministic:
        # DQN.predict 每次调用都现读这个属性，改写它就从下一帧起生效
        model.exploration_rate = playthrough.rate
    state, _ = env.reset()
    with VideoRecorder(playthrough.video_path, fps=config.play.video_fps) as recorder:
        total_reward, step, flag_get, viewer_closed = _run_episode(
            env, model, recorder, state, viewer, deterministic=deterministic
        )
        logger.info(
            "关卡 %s%s：探索率=%s，步数=%d，奖励=%.1f，通关=%s，已录制 %d 帧 → %s",
            playthrough.level_label, playthrough.progress,
            _rate_label(playthrough.rate), step, total_reward, flag_get,
            recorder.frame_count, playthrough.shown_path,
        )
    return viewer_closed


def _log_level_start(level, level_idx, level_count, runs, video_dir) -> None:
    logger.info(
        "=== 播放关卡 %s（%d/%d），共 %d 遍 → %s ===",
        level.label, level_idx, level_count, runs, video_dir,
    )


def _timestamped_dir(base: Path) -> Path:
    """在 base 下开一个以本地时间命名的新目录，让每次播放各存一份。

    没有它的话，同一模型反复播放会一直覆写同名文件，上一场的画面就没了。

    同一秒内再播一场（测试里连跑两次也会）时追加序号，绝不写进已有的那次——
    这个目录的全部意义就是不覆盖。
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base / f"{stamp}_{suffix}"
    return candidate


class PlaybackSession:
    """一场播放的编排：协作者在构造期注入，复杂度全部收进本类。

    默认协作者就是生产实现（DQN 加载、马里奥环境、matplotlib 窗口、核数一半
    的线程额度），直接 ``PlaybackSession().run(config)`` 即真实播放；测试在
    同一个接缝上传假件，不必 patch 模块全局。

    ``run`` 不吞 KeyboardInterrupt：命令行入口要消化它（见 play），嵌在训练里
    调用时要让它冒泡去停训练，两种诉求都不该写进编排本身。
    """

    def __init__(
        self,
        load_model=None,
        make_env=None,
        open_viewer=None,
        thread_count=None,
    ):
        # make_mario_env 的签名 (config, render_mode, level_name) 天然就是这里
        # 要的形状，直接当默认协作者；render_mode 由 run() 强制传录制模式
        self._load_model = load_model or (lambda path: DQN.load(str(path), device="auto"))
        self._make_env = make_env or make_mario_env
        self._open_viewer_fn = open_viewer or _open_viewer
        self._thread_count = thread_count or _playback_threads
        # 每个播放线程自己的模型。DQN.predict 的 ε-greedy 要读
        # model.exploration_rate，多个线程共用一个实例就会互相改写这个属性，
        # 视频目录名立刻变成假话。自备一份副本的代价只是几 MB 权重和一次
        # 加载，换来的是线程之间零共享可变状态。
        self._worker_state = threading.local()
        # run() 一开始就会填上；薄入口用它把中断日志里的目录说清楚
        self.video_dir: Path | None = None

    def run(self, config: AppConfig) -> None:
        """跑完整场播放：校验 checkpoint、抽样、排计划、串行或并发地录完。

        使用训练好的 SB3 DQN 模型播放游戏：一边实时显示，一边录成视频。每个
        关卡输出一个 mp4，默认放在 ``save_root/videos/<时间戳>`` 下——每播一次
        新开一个目录，好让同一模型的多场播放能并排比较。用 ``--no-show`` 可
        只录制不弹窗；没有显示器的环境（服务器 / SSH）会自动走这个行为。

        默认每关贪心地跑一遍。把 ``config.play.sample_runs`` 设成 N，则抽 N
        个探索率，每个概率把配置里的全部关卡各播一遍，各录一个视频，落在以
        该探索率命名的子目录下（``<时间戳>/0-041/1-1.mp4``）——贪心配固定初始
        状态时每次播放都逐帧相同，只有采样才能让轨迹分岔。

        没有显示窗口时各趟播放摊到多个线程上并发跑，线程数取 CPU 核心数的
        一半；有窗口就得一趟一趟地喂帧，快不了。停止方式：关掉显示窗口，或
        由调用方（play 薄入口）响应 Ctrl+C——两者都保住已录制的部分。
        """
        checkpoint_path = config.artifacts.checkpoint
        if checkpoint_path is None or not checkpoint_path.exists():
            logger.error(f"错误：checkpoint 不存在（{checkpoint_path}）")
            return

        levels = config.env.levels or [None]
        sample_runs = config.play.sample_runs
        # 没开采样就是贪心地跑一遍；两者共用同一套计划，省得代码里到处分岔
        runs = sample_runs or 1
        deterministic = sample_runs == 0
        base_dir = config.play.video_dir or (config.artifacts.save_root / "videos")
        # 训练侧自己按步数分好了目录，再来一层时间戳只是多余的套娃
        video_dir = _timestamped_dir(base_dir) if config.play.timestamp_videos else base_dir
        self.video_dir = video_dir
        logger.info("视频输出目录: %s", video_dir)

        # 有窗口才需要主线程这份模型喂画面；没有窗口时模型由各播放线程自备，
        # 只在要读 checkpoint 的探索率时才非加载不可
        viewer = self._open_viewer_fn(config.play)
        model = None
        # 每一遍要用的探索率，开跑前一次性抽好，配置里的全部关卡共用同一个——
        # 于是每个概率对应一个目录，装下这个随机强度下各关卡分别打成什么样。
        # 贪心播放不抽样，占位一个 None 让按遍号取用时不必分岔
        rates: list = [None]

        try:
            if viewer is not None or sample_runs:
                logger.info(f"正在加载 checkpoint: {checkpoint_path}")
                model = self._load_model(checkpoint_path)
                logger.info(f"模型已加载。已训练步数: {model.num_timesteps}")
                if sample_runs:
                    # 必须在 load 之后：DQN.load 会拿存档里的 seed 重播 numpy，
                    # 在那之前重新播种等于没播
                    _reseed_for_sampling()
                    # checkpoint 里的探索率是抽样的均值，可能是 0.02 也可能是训练
                    # 中途的 1.0。打出来，免得画面糊了还以为是模型的问题
                    base_rate = float(getattr(model, "exploration_rate", float("nan")))
                    if _has_exploration(base_rate):
                        rates = [_draw_exploration_rate(base_rate) for _ in range(runs)]
                        logger.info(
                            "按 ε-greedy 采样动作玩 %d 遍：每遍在 [0, %.3f] 上各抽一个"
                            "探索率（均值 %.3f），这一批抽到 %s",
                            runs, 2 * base_rate, base_rate,
                            "、".join(_rate_label(rate) for rate in rates),
                        )
                    else:
                        # 抽不出随机性：每一遍都退回 checkpoint 原值（也就是 0 或
                        # nan）。长度仍与遍数一致，按遍号取用不会越界
                        rates = [base_rate] * runs
                        logger.warning(
                            "checkpoint 的探索率是 %s（缺失或已衰减到 0），抽不出任何"
                            "随机性，每遍都会退化成贪心、彼此逐帧相同", base_rate,
                        )

            plan = _plan_playthroughs(video_dir, levels, config.env, runs, rates)
            if viewer is None:
                self._play_parallel(plan, config, checkpoint_path, deterministic, video_dir)
            else:
                self._play_serially(plan, config, model, viewer, deterministic, video_dir)
        finally:
            if viewer is not None:
                viewer.close()

    def _play_serially(self, plan, config, model, viewer, deterministic, video_dir) -> None:
        """有显示窗口时一趟一趟地播：画面必须按顺序送进同一个窗口，快不了。

        同一个关卡的各遍复用同一个环境：建一个马里奥环境要一百多毫秒，而 reset()
        本来就会把游戏打回初始状态。
        """
        for level_idx, level in enumerate(plan, start=1):
            _log_level_start(level, level_idx, len(plan), level.playthroughs[0].runs, video_dir)
            env = self._make_env(
                config.env, render_mode=RECORD_RENDER_MODE, level_name=level.level_name
            )
            try:
                for playthrough in level.playthroughs:
                    closed = _record_playthrough(
                        playthrough, env, model, config, deterministic, viewer
                    )
                    if closed:
                        logger.info(
                            "显示窗口已关闭，停止播放；已录制的视频保留在 %s", video_dir
                        )
                        return
            finally:
                env.close()

    def _worker_model(self, checkpoint_path, deterministic):
        """取本线程的模型，没有就现加载一份。

        缓存按 checkpoint 区分：同一个会话先后跑两份不同的模型时，绝不能把
        第一份的权重喂给第二场的视频。DQN.load 会拿存档里的 seed 重播全局
        numpy。采样播放时这会把主线程刚按熵播下的种按回固定值，各线程的采样
        序列因此彼此撞车——加载完立刻交还随机性。贪心播放一个随机数都不抽，
        就别去动随机流了：训练侧录最优模型是在训练进程里跑的，平白重播
        numpy 会搅乱训练自身的随机性。
        """
        cache_key = str(checkpoint_path)
        cached = getattr(self._worker_state, "model", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        model = self._load_model(checkpoint_path)
        if not deterministic:
            _reseed_for_sampling()
        self._worker_state.model = (cache_key, model)
        return model

    def _play_one(self, playthrough, config, checkpoint_path, deterministic) -> None:
        """一个线程里跑完一趟：自备模型与环境，录完就收工。

        环境按趟建而不是按关卡复用：趟次怎么分给线程由调度决定，缓存环境就得处理
        跨线程的归属和收尾，为省一百多毫秒不值当。
        """
        env = self._make_env(
            config.env, render_mode=RECORD_RENDER_MODE, level_name=playthrough.level_name
        )
        try:
            model = self._worker_model(checkpoint_path, deterministic)
            _record_playthrough(playthrough, env, model, config, deterministic)
        finally:
            env.close()

    def _play_parallel(self, plan, config, checkpoint_path, deterministic, video_dir) -> None:
        """没有显示窗口时把各趟播放摊到线程上并发跑。"""
        playthroughs = [pt for level in plan for pt in level.playthroughs]
        for level_idx, level in enumerate(plan, start=1):
            _log_level_start(level, level_idx, len(plan), level.playthroughs[0].runs, video_dir)
        workers = min(self._thread_count(), len(playthroughs))
        # 只有一趟时按额度也只会开一个线程，说"并发执行"名不副实，不如不说
        if workers > 1:
            logger.info(
                "无显示窗口，%d 趟播放分给 %d 个线程并发执行（每线程各加载一份模型）",
                len(playthroughs), workers,
            )
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mario-play")
        try:
            # 逐个取结果而不是等 as_completed：线程里冒出来的异常要带回主线程，
            # 否则一个坏掉的环境会静默地少录几段视频
            for future in [pool.submit(self._play_one, pt, config, checkpoint_path,
                                       deterministic)
                           for pt in playthroughs]:
                future.result()
        finally:
            # cancel_futures 只拦得住还没开跑的趟次；线程没法从外面掐断，已经在播的
            # 会自己收尾，所以中断后最多还要等 workers 趟播完
            pool.shutdown(wait=True, cancel_futures=True)


def play(config: AppConfig, session: PlaybackSession | None = None) -> None:
    """命令行播放入口：消化 Ctrl+C（只记一条日志，不打堆栈），其余委托会话。

    嵌在训练里的调用不走这里——直接构造 PlaybackSession 并 run，让中断照常
    冒泡去停训练；"吞不吞中断"是入口的事，不是编排的事。session 仅供测试
    在同一个接缝上注入假协作者，命令行调用不传。
    """
    session = session or PlaybackSession()
    try:
        session.run(config)
    except KeyboardInterrupt:
        # Ctrl+C 是命令行播放唯一的停止方式，不该打出整页堆栈
        logger.info("播放被中断，已录制的视频保留在 %s", session.video_dir)
