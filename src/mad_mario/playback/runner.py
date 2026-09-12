"""加载一个 checkpoint，跑完各关卡并把画面录成视频。

放在 playback 下而不是 cli 里，是为了让训练侧（best 模型刷新时自动录一段）
也能复用：trainer → playback.runner 是单向依赖，不会和 cli → trainer 成环。
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from stable_baselines3 import DQN

from mad_mario.config import AppConfig
from mad_mario.env.factory import make_mario_env
from mad_mario.playback.video import VideoRecorder
from mad_mario.playback.viewer import FrameViewer, is_gui_available

logger = logging.getLogger(__name__)

# 录制只认 rgb_array。human 模式会弹出 pygame 窗口并让 render() 返回 None，
# 与"播放不弹窗"的目标相悖——所以这里覆盖配置里的 render_mode，而不是照搬。
RECORD_RENDER_MODE = "rgb_array"


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

    模式已由 play() 强制为 RECORD_RENDER_MODE，正常不会拿到空帧；这里兜底
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


def _run_episode(env, model, recorder, state, viewer=None):
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

        # 贪心动作
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        step += 1

        # 先录帧再判结束，这样终局画面也在视频里
        _emit_frame(env, recorder, viewer)

        if terminated or truncated or info.get("flag_get"):
            flag_get = bool(info.get("flag_get", False))
            break

    return total_reward, step, flag_get, False


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


def play(config: AppConfig, catch_interrupt: bool = True) -> None:
    """使用训练好的 SB3 DQN 模型播放游戏：一边实时显示，一边录成视频。

    每个关卡输出一个 mp4，默认放在 ``save_root/videos/<时间戳>`` 下——每播一次
    新开一个目录，好让同一模型的多场播放能并排比较。用 ``--no-show`` 可只录制
    不弹窗；没有显示器的环境（服务器 / SSH）会自动走这个行为。

    停止方式：关掉显示窗口，或按 Ctrl+C——两者都会保住已录制的部分。

    Args:
        catch_interrupt: 为 True（命令行播放入口）时自行消化 Ctrl+C，只记一条
            日志；嵌在训练里调用时应传 False，让中断照常冒泡去停训练。
    """
    checkpoint_path = config.artifacts.checkpoint
    if checkpoint_path is None or not checkpoint_path.exists():
        logger.error(f"错误：checkpoint 不存在（{checkpoint_path}）")
        return

    levels = config.env.levels or [None]
    base_dir = config.play.video_dir or (config.artifacts.save_root / "videos")
    # 训练侧自己按步数分好了目录，再来一层时间戳只是多余的套娃
    video_dir = _timestamped_dir(base_dir) if config.play.timestamp_videos else base_dir
    logger.info("视频输出目录: %s", video_dir)
    model = None
    viewer = _open_viewer(config.play)

    try:
        for level_idx, level_name in enumerate(levels, start=1):
            env = make_mario_env(
                config.env, render_mode=RECORD_RENDER_MODE, level_name=level_name
            )

            # 首次加载模型
            if model is None:
                logger.info(f"正在加载 checkpoint: {checkpoint_path}")
                model = DQN.load(str(checkpoint_path), device="auto")
                logger.info(f"模型已加载。已训练步数: {model.num_timesteps}")

            level_label = level_name or f"世界{config.env.world}-{config.env.level}"
            video_path = video_dir / f"{_level_short_name(level_name, config.env)}.mp4"
            logger.info(
                "=== 播放关卡 %s（%d/%d）→ %s ===",
                level_label, level_idx, len(levels), video_path,
            )

            viewer_closed = False
            try:
                state, _ = env.reset()
                with VideoRecorder(video_path, fps=config.play.video_fps) as recorder:
                    total_reward, step, flag_get, viewer_closed = _run_episode(
                        env, model, recorder, state, viewer
                    )
                    logger.info(
                        "关卡 %s：步数=%d，奖励=%.1f，通关=%s，已录制 %d 帧 → %s",
                        level_label, step, total_reward, flag_get,
                        recorder.frame_count, video_path,
                    )
            finally:
                env.close()

            if viewer_closed:
                logger.info("显示窗口已关闭，停止播放；已录制的视频保留在 %s", video_dir)
                break
    except KeyboardInterrupt:
        if not catch_interrupt:
            raise
        # Ctrl+C 是命令行播放唯一的停止方式，不该打出整页堆栈
        logger.info("播放被中断，已录制的视频保留在 %s", video_dir)
    finally:
        if viewer is not None:
            viewer.close()
