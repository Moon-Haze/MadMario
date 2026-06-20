import datetime
from collections import deque
from pathlib import Path

import numpy as np
from tqdm import tqdm

from agent import Mario
from config import TrainingConfig, resolve_checkpoint
from env_factory import make_mario_env, make_vector_env
from metrics import MetricLogger


def create_save_dir(config: TrainingConfig):
    save_dir = config.save_root / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def announce_checkpoint(checkpoint):
    if checkpoint:
        tqdm.write(f"检测到检查点：{checkpoint}，将自动恢复训练。")
    else:
        tqdm.write("未检测到检查点，将从头开始训练。")


def rolling_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


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


def build_mario(config: TrainingConfig, action_dim, save_dir):
    checkpoint = resolve_checkpoint(config)
    announce_checkpoint(checkpoint)
    return Mario(
        state_dim=config.state_dim,
        action_dim=action_dim,
        save_dir=save_dir,
        checkpoint=checkpoint,
        config=config,
    )


def train_single_env(config: TrainingConfig):
    env = make_mario_env()
    save_dir = create_save_dir(config)
    mario = build_mario(config, env.action_space.n, save_dir)
    completed_episodes = mario.loaded_episode
    logger = MetricLogger(mario.save_dir)

    progress_bar = tqdm(
        total=config.episodes,
        initial=completed_episodes,
        desc="训练进度",
        unit="回合",
        position=0,
    )
    status_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=False)

    try:
        while completed_episodes < config.episodes:
            state, _ = env.reset()
            ep_reward = 0.0
            ep_length = 0
            last_loss = None
            last_q = None

            while True:
                action = mario.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                mario.cache(state, next_state, action, reward, done)
                q, loss = mario.learn()
                logger.log_step(reward, loss, q)
                ep_reward += reward
                ep_length += 1
                if loss is not None:
                    last_loss = loss
                    last_q = q

                state = next_state
                if done or info.get("flag_get"):
                    break

            completed_episodes += 1
            mario.current_episode = completed_episodes
            logger.log_episode()
            progress_bar.update(1)
            status_bar.set_description_str(
                f"步数={mario.curr_step} | "
                f"探索率={mario.exploration_rate:.3f} | "
                f"回合奖励={ep_reward:.1f} | "
                f"回合长度={ep_length} | "
                f"损失={'-' if last_loss is None else f'{last_loss:.4f}'} | "
                f"Q值={'-' if last_q is None else f'{last_q:.4f}'}"
            )

            if completed_episodes % config.record_every == 0:
                logger.record(
                    episode=completed_episodes,
                    epsilon=mario.exploration_rate,
                    step=mario.curr_step,
                )
                mario.save(episode=completed_episodes)

        mario.save(episode=completed_episodes)
    except KeyboardInterrupt:
        tqdm.write("检测到训练中断，正在保存最新检查点...")
        mario.save(episode=mario.current_episode)
        raise
    finally:
        status_bar.close()
        progress_bar.close()
        env.close()


def train_vector_env(config: TrainingConfig):
    envs = make_vector_env(config.num_envs)
    states, _ = envs.reset()
    save_dir = create_save_dir(config)
    mario = build_mario(config, envs.single_action_space.n, save_dir)
    completed_episodes = mario.loaded_episode
    logger = MetricLogger(mario.save_dir)

    ep_rewards = np.zeros(config.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(config.num_envs, dtype=np.int32)
    recent_losses = deque(maxlen=100)
    recent_qs = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)
    last_loss = None
    last_q = None

    progress_bar = tqdm(
        total=config.episodes,
        initial=completed_episodes,
        desc="并行训练进度",
        unit="回合",
        position=0,
    )
    status_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=False)

    try:
        while completed_episodes < config.episodes:
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
                recent_losses.append(loss)
                recent_qs.append(q)

            for env_index, done in enumerate(dones):
                if not done:
                    continue

                completed_episodes += 1
                mario.current_episode = completed_episodes
                episode_reward = float(ep_rewards[env_index])
                recent_rewards.append(episode_reward)
                logger.log_episode_metrics(
                    episode_reward,
                    int(ep_lengths[env_index]),
                    rolling_mean(recent_losses),
                    rolling_mean(recent_qs),
                )
                progress_bar.update(1)

                ep_rewards[env_index] = 0.0
                ep_lengths[env_index] = 0

                if completed_episodes % config.record_every == 0:
                    logger.record(
                        episode=completed_episodes,
                        epsilon=mario.exploration_rate,
                        step=mario.curr_step,
                    )
                    mario.save(episode=completed_episodes)

                if completed_episodes >= config.episodes:
                    break

            status_bar.set_description_str(
                f"环境数={config.num_envs} | "
                f"步数={mario.curr_step} | "
                f"探索率={mario.exploration_rate:.3f} | "
                f"近100回合奖励={rolling_mean(recent_rewards):.1f} | "
                f"损失={'-' if last_loss is None else f'{last_loss:.4f}'} | "
                f"Q值={'-' if last_q is None else f'{last_q:.4f}'}"
            )
            states = next_states

        mario.save(episode=completed_episodes)
    except KeyboardInterrupt:
        tqdm.write("检测到训练中断，正在保存最新检查点...")
        mario.save(episode=mario.current_episode)
        raise
    finally:
        status_bar.close()
        progress_bar.close()
        envs.close()
