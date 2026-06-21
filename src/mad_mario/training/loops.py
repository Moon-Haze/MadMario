import warnings

import numpy as np
from tqdm import tqdm

from mad_mario.training.evaluation import evaluate_agent
from mad_mario.training.progress import TrainingProgress, recent_metric_buffers, rolling_mean


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


def should_record(episode, record_every):
    return episode > 0 and episode % record_every == 0


def should_eval(episode, eval_every):
    return eval_every > 0 and episode > 0 and episode % eval_every == 0


def run_evaluation(agent, checkpoint_manager, config, best_eval_score):
    result = evaluate_agent(agent, config)
    tqdm.write(
        f"评估结果 | 分数={result.score:.3f} | 平均奖励={result.mean_reward:.3f} | "
        f"平均长度={result.mean_length:.3f} | 平均最大 x={result.mean_max_x_pos:.3f} | 通关率={result.flag_rate:.3f}"
    )
    if result.score > best_eval_score:
        best_eval_score = result.score
        checkpoint_manager.save_best(
            agent,
            eval_score=result.score,
            eval_reward=result.mean_reward,
            eval_max_x_pos=result.mean_max_x_pos,
            eval_flag_rate=result.flag_rate,
            episode=agent.current_episode,
        )
    return best_eval_score


def should_save_step(agent, next_save_step):
    return agent.curr_step >= next_save_step


def train_single_env_loop(env, agent, logger, checkpoint_manager, config):
    completed_episodes = agent.loaded_episode
    next_save_step = _next_interval_step(agent.curr_step, config.artifacts.save_every)
    progress = TrainingProgress(config.training.episodes, initial=completed_episodes, desc="训练进度")
    best_eval_score = float("-inf")
    interrupted = False

    try:
        while completed_episodes < config.training.episodes:
            state, _ = env.reset()
            ep_reward = 0.0
            ep_length = 0
            ep_max_x_pos = 0.0
            action_counts = np.zeros(agent.action_dim, dtype=np.int32)
            last_loss = None
            last_q = None

            while True:
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                action_counts[action] += 1
                agent.cache(state, next_state, action, reward, done)
                q, loss = agent.learn()
                logger.log_step(reward, loss, q)
                ep_reward += reward
                ep_length += 1
                ep_max_x_pos = max(ep_max_x_pos, _info_float(info, "max_x_pos", _info_float(info, "x_pos", 0.0)))
                if loss is not None:
                    last_loss = loss
                    last_q = q

                if should_save_step(agent, next_save_step):
                    checkpoint_manager.save(agent)
                    next_save_step = _next_interval_step(agent.curr_step, config.artifacts.save_every)

                state = next_state
                if done or info.get("flag_get"):
                    break

            completed_episodes += 1
            agent.current_episode = completed_episodes
            noop_rate = float(action_counts[0]) / max(1, ep_length)
            most_used_action = int(np.argmax(action_counts))
            logger.log_episode(max_x_pos=ep_max_x_pos, noop_rate=noop_rate, most_used_action=most_used_action)
            progress.update(1)
            progress.set_single_status(agent, ep_reward, ep_length, last_loss, last_q)

            emit_record = should_record(completed_episodes, config.training.record_every)
            logger.record(
                episode=completed_episodes,
                epsilon=agent.exploration_rate,
                step=agent.curr_step,
                emit=emit_record,
            )
            if emit_record:
                checkpoint_manager.save(agent, episode=completed_episodes)

            if should_eval(completed_episodes, config.training.eval_every):
                best_eval_score = run_evaluation(agent, checkpoint_manager, config, best_eval_score)

        checkpoint_manager.save(agent, episode=completed_episodes)
    except KeyboardInterrupt:
        interrupted = True
        tqdm.write("检测到训练中断，跳过保存检查点...")
    finally:
        progress.close()
        env.close()

    if interrupted:
        tqdm.write("训练已中断，资源已清理。")


def train_vector_env_loop(envs, states, agent, logger, checkpoint_manager, config):
    completed_episodes = agent.loaded_episode
    next_save_step = _next_interval_step(agent.curr_step, config.artifacts.save_every)
    ep_rewards = np.zeros(config.training.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(config.training.num_envs, dtype=np.int32)
    ep_max_x_positions = np.zeros(config.training.num_envs, dtype=np.float32)
    ep_action_counts = np.zeros((config.training.num_envs, agent.action_dim), dtype=np.int32)
    recent_losses, recent_qs, recent_rewards = recent_metric_buffers()
    last_loss = None
    last_q = None
    progress = TrainingProgress(config.training.episodes, initial=completed_episodes, desc="并行训练进度")
    best_eval_score = float("-inf")
    interrupted = False

    try:
        while completed_episodes < config.training.episodes:
            actions = agent.act_batch(states)
            next_states, rewards, terminated, truncated, infos = envs.step(actions)
            dones = np.logical_or(terminated, truncated)
            replay_next_states = get_replay_next_states(next_states, infos, dones)

            agent.cache_batch(states, replay_next_states, actions, rewards, dones)
            ep_rewards += rewards
            ep_lengths += 1
            ep_max_x_positions = np.maximum(ep_max_x_positions, get_info_array(infos, "max_x_pos", get_info_array(infos, "x_pos", 0.0)))
            for env_index, act in enumerate(actions):
                ep_action_counts[env_index, act] += 1

            q, loss = agent.learn()
            if loss is not None:
                last_loss = loss
                last_q = q
                recent_losses.append(loss)
                recent_qs.append(q)

            if should_save_step(agent, next_save_step):
                checkpoint_manager.save(agent)
                next_save_step = _next_interval_step(agent.curr_step, config.artifacts.save_every)

            for env_index, done in enumerate(dones):
                if not done:
                    continue

                completed_episodes += 1
                agent.current_episode = completed_episodes
                episode_reward = float(ep_rewards[env_index])
                recent_rewards.append(episode_reward)
                ep_length = int(ep_lengths[env_index])
                noop_rate = float(ep_action_counts[env_index, 0]) / max(1, ep_length)
                most_used_action = int(np.argmax(ep_action_counts[env_index]))
                logger.log_episode_metrics(
                    episode_reward,
                    ep_length,
                    rolling_mean(recent_losses),
                    rolling_mean(recent_qs),
                    float(ep_max_x_positions[env_index]),
                    noop_rate,
                    most_used_action,
                )
                progress.update(1)

                ep_rewards[env_index] = 0.0
                ep_lengths[env_index] = 0
                ep_max_x_positions[env_index] = 0.0
                ep_action_counts[env_index] = 0

                emit_record = should_record(completed_episodes, config.training.record_every)
                logger.record(
                    episode=completed_episodes,
                    epsilon=agent.exploration_rate,
                    step=agent.curr_step,
                    emit=emit_record,
                )
                if emit_record:
                    checkpoint_manager.save(agent, episode=completed_episodes)

                if should_eval(completed_episodes, config.training.eval_every):
                    best_eval_score = run_evaluation(agent, checkpoint_manager, config, best_eval_score)

                if completed_episodes >= config.training.episodes:
                    break

            progress.set_vector_status(config.training, agent, recent_rewards, last_loss, last_q)
            states = next_states

        checkpoint_manager.save(agent, episode=completed_episodes)
    except KeyboardInterrupt:
        interrupted = True
        tqdm.write("检测到训练中断，跳过保存检查点...")
    finally:
        progress.close()
        close_vector_envs(envs, terminate=interrupted)

    if interrupted:
        tqdm.write("训练已中断，资源已清理。")


def get_info_array(infos, key, default):
    values = infos.get(key)
    if values is None:
        return default
    return np.asarray(values, dtype=np.float32)


def _info_float(info, key, default):
    try:
        return float(info.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def close_vector_envs(envs, terminate=False):
    if not terminate:
        envs.close()
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Calling `close` while waiting for a pending call.*",
            category=UserWarning,
        )
        envs.close(terminate=True)


def _next_interval_step(current_step, interval):
    return ((int(current_step) // interval) + 1) * interval
