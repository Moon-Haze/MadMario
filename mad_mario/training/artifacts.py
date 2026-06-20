from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

from mad_mario.config import ArtifactConfig


@dataclass
class RunArtifacts:
    save_root: Path
    latest_checkpoint: Path
    latest_metrics_csv: Path
    latest_plot_paths: list[Path]
    run_dir: Path | None = None
    run_checkpoint: Path | None = None
    run_metrics_csv: Path | None = None
    run_plot_paths: list[Path] | None = None

    @property
    def metrics_csv_paths(self) -> list[Path]:
        paths = [self.latest_metrics_csv]
        if self.run_metrics_csv is not None:
            paths.append(self.run_metrics_csv)
        return paths

    @property
    def plot_path_groups(self) -> list[list[Path]]:
        groups = [self.latest_plot_paths]
        if self.run_plot_paths is not None:
            groups.append(self.run_plot_paths)
        return groups

    @property
    def checkpoint_paths(self) -> list[Path]:
        paths = [self.latest_checkpoint]
        if self.run_checkpoint is not None:
            paths.append(self.run_checkpoint)
        return paths


def create_artifacts(config: ArtifactConfig) -> RunArtifacts:
    save_root = config.save_root
    save_root.mkdir(parents=True, exist_ok=True)

    latest_plot_paths = [
        save_root / "latest_reward_plot.png",
        save_root / "latest_length_plot.png",
        save_root / "latest_loss_plot.png",
        save_root / "latest_q_plot.png",
    ]

    run_dir = None
    run_checkpoint = None
    run_metrics_csv = None
    run_plot_paths = None
    if config.keep_runs:
        run_name = config.run_name or datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = save_root / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_checkpoint = run_dir / "checkpoint.chkpt"
        run_metrics_csv = run_dir / "metrics.csv"
        run_plot_paths = [
            run_dir / "reward_plot.png",
            run_dir / "length_plot.png",
            run_dir / "loss_plot.png",
            run_dir / "q_plot.png",
        ]

    return RunArtifacts(
        save_root=save_root,
        latest_checkpoint=save_root / "latest.chkpt",
        latest_metrics_csv=save_root / "latest_metrics.csv",
        latest_plot_paths=latest_plot_paths,
        run_dir=run_dir,
        run_checkpoint=run_checkpoint,
        run_metrics_csv=run_metrics_csv,
        run_plot_paths=run_plot_paths,
    )
