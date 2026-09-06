import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# 限制 BLAS 线程数防止多进程环境下的线程爆炸（每个子进程 × 自动线程数 >= ULIMIT）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from mad_mario.config import (
    _apply_config_defaults,
    _extract_config_path,
    build_parser,
    config_from_play_args,
    config_from_train_args,
    load_config_file,
)
from mad_mario.env.factory import make_mario_env
from mad_mario.logging.setup import setup_logging
from mad_mario.training.trainer import train

logger = logging.getLogger(__name__)


def main():
    # 1. 提前扫描 --config（不经过 argparse，因为需要在解析前获得配置）
    config_path = _extract_config_path()
    config = {}
    if config_path:
        config = load_config_file(config_path)

    # 2. 构建解析器
    parser = build_parser()

    # 3. 将配置项作为默认值注入所有子解析器
    if config:
        _apply_config_defaults(parser, config)

    # 4. 正常解析——CLI 参数自然覆盖 config 中的对应值
    args = parser.parse_args()

    if args.command == "train":
        config = config_from_train_args(args)
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        log_path = setup_logging(level=log_level, log_dir=config.artifacts.save_root / "logs")
        train(config, log_path=log_path)
    elif args.command == "play":
        setup_logging()
        play(config_from_play_args(args))
    else:
        parser.error(f"未知命令: {args.command}")


def play(config):
    """使用训练好的 SB3 DQN 模型播放游戏。"""
    checkpoint_path = config.artifacts.checkpoint
    if checkpoint_path is None or not checkpoint_path.exists():
        logger.error(f"错误：checkpoint 不存在（{checkpoint_path}）")
        return

    levels = config.env.levels or [None]
    model = None
    fig = None
    image = None
    ax = None

    for level_idx, level_name in enumerate(levels, start=1):
        env = make_mario_env(config.env, level_name=level_name)

        # 首次加载模型
        if model is None:
            logger.info(f"正在加载 checkpoint: {checkpoint_path}")
            model = DQN.load(str(checkpoint_path), device="auto")
            logger.info(f"模型已加载。已训练步数: {model.num_timesteps}")

            plt.ion()
            fig, ax = plt.subplots()
            ax.axis("off")
            fig.canvas.manager.set_window_title("训练好的 Mario (SB3 DQN)")  # type: ignore[union-attr]

        level_label = level_name or f"世界{config.env.world}-{config.env.level}"
        logger.info("=== 播放关卡 %s（%d/%d）===", level_label, level_idx, len(levels))

        try:
            state, _ = env.reset()
            total_reward = 0.0
            step = 0
            flag_get = False

            while True:
                if not plt.get_fignums():
                    break

                # 渲染
                frame = env.render()
                if image is None:
                    image = ax.imshow(frame)  # type: ignore[union-attr]
                else:
                    image.set_data(frame)
                fig.canvas.draw_idle()  # type: ignore[union-attr]
                fig.canvas.flush_events()  # type: ignore[union-attr]
                plt.pause(0.001)

                # 贪心动作
                action, _ = model.predict(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated

                total_reward += float(reward)
                step += 1

                if done or info.get("flag_get"):
                    flag_get = bool(info.get("flag_get", False))
                    logger.info(
                        "关卡 %s：步数=%d，奖励=%.1f，通关=%s",
                        level_label, step, total_reward, flag_get,
                    )
                    break

            if not plt.get_fignums():
                break

        finally:
            env.close()

    if fig is not None:
        plt.ioff()
        plt.close(fig)


if __name__ == "__main__":
    main()
