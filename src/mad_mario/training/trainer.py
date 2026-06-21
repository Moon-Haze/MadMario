from mad_mario.agent.checkpoint import CheckpointManager
from mad_mario.agent.mario import Mario
from mad_mario.config import AppConfig
from mad_mario.env.factory import make_mario_env, make_vector_env
from mad_mario.logging.metrics import MetricLogger
from mad_mario.training.artifacts import create_artifacts
from mad_mario.training.loops import train_single_env_loop, train_vector_env_loop


def build_agent(config: AppConfig, action_dim):
    return Mario(
        state_dim=config.env.state_dim,
        action_dim=action_dim,
        agent_config=config.agent,
        training_config=config.training,
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
    env = make_mario_env(config.env)
    agent = build_agent(config, env.action_space.n)
    checkpoint_manager = CheckpointManager(config.artifacts, artifacts)
    loaded_checkpoint = checkpoint_manager.load_if_available(agent)
    logger = MetricLogger(
        artifacts.metrics_csv_paths,
        artifacts.plot_path_groups,
        reset=loaded_checkpoint is None,
        tensorboard_dir=artifacts.save_root / "tensorboard",
    )
    train_single_env_loop(env, agent, logger, checkpoint_manager, config)


def train_vector(config: AppConfig, artifacts):
    envs = make_vector_env(config.training.num_envs, config.env)
    states, _ = envs.reset()
    agent = build_agent(config, envs.single_action_space.n)
    checkpoint_manager = CheckpointManager(config.artifacts, artifacts)
    loaded_checkpoint = checkpoint_manager.load_if_available(agent)
    logger = MetricLogger(
        artifacts.metrics_csv_paths,
        artifacts.plot_path_groups,
        reset=loaded_checkpoint is None,
        tensorboard_dir=artifacts.save_root / "tensorboard",
    )
    train_vector_env_loop(envs, states, agent, logger, checkpoint_manager, config)
