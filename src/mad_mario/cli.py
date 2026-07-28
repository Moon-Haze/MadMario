import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# 限制 BLAS 线程数防止多进程环境下的线程爆炸（每个子进程 × 自动线程数 >= ULIMIT）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from mad_mario.config import build_parser, config_from_play_args, config_from_train_args
from mad_mario.env.factory import make_mario_env
from mad_mario.training.trainer import train


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train(config_from_train_args(args))
    elif args.command == "play":
        play(config_from_play_args(args))
    else:
        parser.error(f"未知命令: {args.command}")


def play(config):
    """使用训练好的 SB3 DQN 模型播放游戏。"""
    checkpoint_path = config.artifacts.checkpoint
    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"错误：checkpoint 不存在（{checkpoint_path}）")
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
            print(f"正在加载 checkpoint: {checkpoint_path}")
            model = DQN.load(str(checkpoint_path), device="auto")
            print(f"模型已加载。已训练步数: {model.num_timesteps}")

            plt.ion()
            fig, ax = plt.subplots()
            ax.axis("off")
            fig.canvas.manager.set_window_title("训练好的 Mario (SB3 DQN)")  # type: ignore[union-attr]

        level_label = level_name or f"世界{config.env.world}-{config.env.level}"
        print(f"\n=== 播放关卡 {level_label}（{level_idx}/{len(levels)}）===")

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
                    print(
                        f"关卡 {level_label}：步数={step}，奖励={total_reward:.1f}，通关={flag_get}"
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
