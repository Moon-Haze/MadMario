import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import datetime
from pathlib import Path

import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import FrameStackObservation as FrameStack, GrayscaleObservation as GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm

from agent import Mario
from metrics import MetricLogger
from wrappers import ResizeObservation, SkipFrame, NormalizeObservation


def make_env():
    def thunk():
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, shape=84)
        env = NormalizeObservation(env)
        env = FrameStack(env, stack_size=4)
        return env
    return thunk


def get_replay_next_states(next_states, infos, dones):
    replay_next_states = np.array(next_states, copy=True)
    final_observations = infos.get("final_observation")
    if final_observations is None:
        return replay_next_states

    final_observation_mask = infos.get("_final_observation", dones)
    for env_index, done in enumerate(dones):
        if not done or not final_observation_mask[env_index]:
            continue
        final_observation = final_observations[env_index]
        if final_observation is not None:
            replay_next_states[env_index] = final_observation
    return replay_next_states


def main():
    num_envs = max(1, (os.cpu_count() or 2) // 2)
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    states, _ = envs.reset()

    save_dir = Path('./checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    # 如果存在最新检查点，则自动恢复训练；否则从头开始训练。
    latest_checkpoint = Path('checkpoints/latest.chkpt')
    checkpoint = latest_checkpoint if latest_checkpoint.exists() else None
    if checkpoint:
        tqdm.write(f"检测到最新检查点：{checkpoint}，将自动恢复训练。")
    else:
        tqdm.write("未检测到最新检查点，将从头开始训练。")

    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=envs.single_action_space.n,
        save_dir=save_dir,
        checkpoint=checkpoint,
    )
    completed_episodes = mario.loaded_episode
    logger = MetricLogger(mario.save_dir)

    episodes = 40000
    ep_rewards = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    last_loss = None
    last_q = None

    progress_bar = tqdm(
        total=episodes,
        initial=completed_episodes,
        desc="并行训练进度",
        unit="回合",
        position=0,
    )
    status_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=False)

    try:
        while completed_episodes < episodes:
            actions = mario.act_batch(states)
            next_states, rewards, terminated, truncated, infos = envs.step(actions)
            dones = np.logical_or(terminated, truncated)
            replay_next_states = get_replay_next_states(next_states, infos, dones)

            mario.cache_batch(states, replay_next_states, actions, rewards, dones)
            ep_rewards += rewards
            ep_lengths += 1

            q, loss = mario.learn()
            if loss is not None:
                last_loss = loss
                last_q = q

            for env_index, done in enumerate(dones):
                if not done:
                    continue

                completed_episodes += 1
                mario.current_episode = completed_episodes
                logger.log_episode_metrics(
                    float(ep_rewards[env_index]),
                    int(ep_lengths[env_index]),
                )
                progress_bar.update(1)

                ep_rewards[env_index] = 0.0
                ep_lengths[env_index] = 0

                if completed_episodes % 20 == 0:
                    logger.record(
                        episode=completed_episodes,
                        epsilon=mario.exploration_rate,
                        step=mario.curr_step,
                    )
                    mario.save(episode=completed_episodes)

                if completed_episodes >= episodes:
                    break

            status_bar.set_description_str(
                f"环境数={num_envs} | "
                f"步数={mario.curr_step} | "
                f"探索率={mario.exploration_rate:.3f} | "
                f"最近奖励={ep_rewards.mean():.1f} | "
                f"损失={'-' if last_loss is None else f'{last_loss:.4f}'} | "
                f"Q值={'-' if last_q is None else f'{last_q:.4f}'}"
            )
            states = next_states
    except KeyboardInterrupt:
        tqdm.write("检测到训练中断，正在保存最新检查点...")
        mario.save(episode=mario.current_episode)
        raise
    finally:
        status_bar.close()
        progress_bar.close()
        envs.close()


if __name__ == "__main__":
    main()
