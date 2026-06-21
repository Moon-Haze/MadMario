from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mad_mario.env.factory import make_mario_env


@dataclass
class EvalResult:
    mean_reward: float
    mean_length: float
    mean_max_x_pos: float
    flag_rate: float
    score: float


def evaluate_agent(agent, config) -> EvalResult:
    env = make_mario_env(config.env)
    rewards = []
    lengths = []
    flags = []
    max_x_positions = []

    try:
        for _ in range(config.training.eval_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_max_x = 0.0
            flag_get = False

            while True:
                action = agent.act_eval(state)
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                flag_get = bool(info.get("flag_get", False))
                episode_max_x = max(episode_max_x, info.get("max_x_pos", info.get("x_pos", 0.0)))

                if done or flag_get:
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)
            flags.append(flag_get)
            max_x_positions.append(episode_max_x)
    finally:
        env.close()

    mean_max_x_pos = float(np.mean(max_x_positions)) if max_x_positions else 0.0
    flag_rate = float(np.mean(flags)) if flags else 0.0

    return EvalResult(
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        mean_max_x_pos=mean_max_x_pos,
        flag_rate=flag_rate,
        score=mean_max_x_pos + 5000.0 * flag_rate,
    )
