import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import matplotlib.pyplot as plt
from tqdm import tqdm

from mad_mario.agent.checkpoint import CheckpointManager
from mad_mario.agent.mario import Mario
from mad_mario.config import build_parser, config_from_play_args, config_from_train_args
from mad_mario.env.factory import make_mario_env
from mad_mario.training.artifacts import create_artifacts
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
    env = make_mario_env(config.env)
    artifacts = create_artifacts(config.artifacts)
    agent = Mario(
        state_dim=config.env.state_dim,
        action_dim=env.action_space.n,
        agent_config=config.agent,
        training_config=config.training,
    )
    CheckpointManager(config.artifacts, artifacts).load(agent, config.artifacts.checkpoint)
    agent.exploration_rate = 0.0
    agent.exploration_rate_min = 0.0

    plt.ion()
    fig, ax = plt.subplots()
    image = None
    ax.axis("off")
    fig.canvas.manager.set_window_title("训练好的 Mario")

    try:
        progress_bar = tqdm(range(1, config.training.episodes + 1), desc="播放进度", unit="回合")
        for episode in progress_bar:
            state, _ = env.reset()
            total_reward = 0.0
            step = 0
            flag_get = False

            while True:
                frame = env.render()
                if image is None:
                    image = ax.imshow(frame)
                else:
                    image.set_data(frame)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

                action = agent.act(state)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward
                step += 1

                if done or info.get("flag_get"):
                    flag_get = bool(info.get("flag_get", False))
                    progress_bar.set_postfix({
                        "步数": step,
                        "奖励": f"{total_reward:.1f}",
                        "通关": flag_get,
                    })
                    tqdm.write(
                        f"第 {episode} 回合结束："
                        f"步数={step}，奖励={total_reward:.1f}，是否通关={flag_get}"
                    )
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
