from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mad_mario.env.factory import make_mario_env


@dataclass
class EvalResult:
    mean_reward: float
    mean_length: float
    flag_rate: float


def evaluate_agent(agent, config) -> EvalResult:
    env = make_mario_env(config.env)
    rewards = []
    lengths = []
    flags = []

    try:
        for _ in range(config.training.eval_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            flag_get = False

            while True:
                action = agent.act_eval(state)
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                flag_get = bool(info.get("flag_get", False))

                if done or flag_get:
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)
            flags.append(flag_get)
    finally:
        env.close()

    return EvalResult(
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        flag_rate=float(np.mean(flags)) if flags else 0.0,
    )
