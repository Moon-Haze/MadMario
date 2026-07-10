from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from mad_mario.agent.checkpoint import CheckpointManager
from mad_mario.agent.mario import Mario
from mad_mario.config import AppConfig
from mad_mario.env.factory import make_mario_env, make_vector_env
from mad_mario.logging.metrics import MetricLogger
from mad_mario.training.artifacts import RunArtifacts, create_artifacts
from mad_mario.training.loops import train_single_env_loop, train_vector_env_loop


@dataclass
class SetupResult:
    """训练启动后的所有组件，供训练循环或 Notebook 使用。"""
    env: Any | None = None          # 单环境时有效
    envs: Any | None = None         # 并行环境时有效
    states: np.ndarray | None = None
    agent: Mario | None = None
    action_dim: int = 0              # 动作空间维度
    checkpoint_manager: CheckpointManager | None = None
    logger: MetricLogger | None = None
    active_levels: list[str] | None = None
    loaded: bool = False


def build_agent(config: AppConfig, action_dim):
    return Mario(
        state_dim=config.env.state_dim,
        action_dim=action_dim,
        agent_config=config.agent,
        training_config=config.training,
    )


def _active_levels(config):
    """返回实际使用的关卡列表，单关卡时返回 None（保持兼容）。"""
    levels = config.env.levels
    if levels and len(levels) > 1:
        return levels
    return None


def setup_training(config: AppConfig, artifacts: RunArtifacts | None = None) -> SetupResult:
    """创建训练所需的所有组件，CLI 和 Notebook 共用此函数。"""
    if artifacts is None:
        artifacts = create_artifacts(config.artifacts)

    loaded = False
    active_levels = _active_levels(config)

    if config.training.vector:
        envs = make_vector_env(config.training.num_envs, config.env)
        states, _ = envs.reset()
        action_dim = envs.single_action_space.n
        agent = build_agent(config, action_dim)
        checkpoint_manager = CheckpointManager(config.artifacts, artifacts)
        loaded = checkpoint_manager.load_if_available(agent) is not None
        logger = MetricLogger(
            artifacts.metrics_csv_paths,
            artifacts.plot_path_groups,
            reset=not loaded,
        )
        return SetupResult(
            envs=envs, states=states,
            agent=agent, action_dim=action_dim,
            checkpoint_manager=checkpoint_manager,
            logger=logger, loaded=loaded,
        )
    else:
        env = make_mario_env(config.env)
        action_dim = env.action_space.n
        agent = build_agent(config, action_dim)
        checkpoint_manager = CheckpointManager(config.artifacts, artifacts)
        loaded = checkpoint_manager.load_if_available(agent) is not None
        logger = MetricLogger(
            artifacts.metrics_csv_paths,
            artifacts.plot_path_groups,
            reset=not loaded,
        )
        return SetupResult(
            env=env, agent=agent, action_dim=action_dim,
            checkpoint_manager=checkpoint_manager,
            logger=logger, active_levels=active_levels, loaded=loaded,
        )


def train(config: AppConfig):
    if config.env.movement != "right_only":
        print("建议先用 --movement right_only 学会稳定向右，再尝试 simple/complex。")
    artifacts = create_artifacts(config.artifacts)
    if config.training.vector:
        train_vector(config, artifacts)
    else:
        train_single(config, artifacts)


def train_single(config: AppConfig, artifacts):
    result = setup_training(config, artifacts)
    train_single_env_loop(
        result.env, result.agent, result.logger,
        result.checkpoint_manager, config, result.active_levels,
    )


def train_vector(config: AppConfig, artifacts):
    result = setup_training(config, artifacts)
    train_vector_env_loop(
        result.envs, result.states, result.agent, result.logger,
        result.checkpoint_manager, config,
    )
