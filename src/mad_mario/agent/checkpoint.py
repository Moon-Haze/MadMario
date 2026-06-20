from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from mad_mario.agent.mario import Mario
from mad_mario.config import ArtifactConfig, resolve_checkpoint
from mad_mario.training.artifacts import RunArtifacts


class CheckpointManager:
    def __init__(self, config: ArtifactConfig, artifacts: RunArtifacts):
        self.config = config
        self.artifacts = artifacts

    def load_if_available(self, agent: Mario) -> Path | None:
        checkpoint_path = resolve_checkpoint(self.config)
        if checkpoint_path is None:
            tqdm.write("未检测到检查点，将从头开始训练。")
            return None

        try:
            self.load(agent, checkpoint_path)
        except Exception as exc:
            if self.config.checkpoint is not None:
                raise
            tqdm.write(f"检查点 {checkpoint_path} 无法加载，将从头开始训练。原因: {exc}")
            return None
        return checkpoint_path

    def load(self, agent: Mario, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise ValueError(f"{checkpoint_path} 不存在")

        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        agent.load_state_dict(checkpoint)
        tqdm.write(
            f"正在加载模型 {checkpoint_path}，"
            f"探索率为 {agent.exploration_rate:.4f}，"
            f"步数为 {agent.curr_step}，"
            f"回合为 {agent.current_episode}"
        )

    def save(self, agent: Mario, episode: int | None = None) -> None:
        if episode is not None:
            agent.current_episode = episode

        checkpoint = agent.state_dict()
        for checkpoint_path in self.artifacts.checkpoint_paths:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)

        tqdm.write(
            f"MarioNet 已保存到 {self.artifacts.latest_checkpoint}，"
            f"当前步数 {agent.curr_step}，当前回合 {agent.current_episode}"
        )

    def save_best(self, agent: Mario, eval_reward: float, episode: int | None = None) -> None:
        if episode is not None:
            agent.current_episode = episode

        checkpoint = agent.state_dict()
        checkpoint["best_eval_reward"] = float(eval_reward)
        for checkpoint_path in self.artifacts.best_checkpoint_paths:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)

        tqdm.write(
            f"最佳 MarioNet 已保存到 {self.artifacts.best_checkpoint}，"
            f"评估奖励 {eval_reward:.3f}，当前步数 {agent.curr_step}，当前回合 {agent.current_episode}"
        )
