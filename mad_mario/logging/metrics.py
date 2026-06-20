import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


METRIC_DECIMALS = 3
EPSILON_DECIMALS = 4
MEAN_WINDOW = 100
PLOT_CONFIGS = [
    ("平均奖励曲线", "记录次数", "平均奖励"),
    ("平均长度曲线", "记录次数", "平均长度"),
    ("平均损失曲线", "记录次数", "平均损失"),
    ("平均 Q 值曲线", "记录次数", "平均 Q 值"),
]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


class MetricLogger:
    def __init__(self, csv_paths: list[Path], plot_path_groups: list[list[Path]], reset: bool = False):
        self.csv_paths = csv_paths
        self.plot_path_groups = plot_path_groups

        if reset:
            self._clear_outputs()
        for csv_path in self.csv_paths:
            self._ensure_csv_header(csv_path)

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.moving_avgs = [[], [], [], []]
        if not reset:
            self._load_history_from_csv(self.csv_paths[0])
        self.init_episode()
        self.record_time = time.time()

    def _clear_outputs(self):
        for csv_path in self.csv_paths:
            if csv_path.exists():
                csv_path.unlink()
        for plot_paths in self.plot_path_groups:
            for plot_path in plot_paths:
                if plot_path.exists():
                    plot_path.unlink()

    def _ensure_csv_header(self, csv_path):
        if csv_path.exists() and csv_path.stat().st_size > 0:
            return
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "step",
                "epsilon",
                "mean_reward",
                "mean_length",
                "mean_loss",
                "mean_q",
                "time_delta",
            ])

    def _load_history_from_csv(self, csv_path):
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    self.moving_avgs[0].append(float(row["mean_reward"]))
                    self.moving_avgs[1].append(float(row["mean_length"]))
                    self.moving_avgs[2].append(float(row["mean_loss"]))
                    self.moving_avgs[3].append(float(row["mean_q"]))
                except (KeyError, TypeError, ValueError):
                    continue

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        """记录一个回合的结束。"""
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = self._round_metric(self.curr_ep_loss / self.curr_ep_loss_length)
            ep_avg_q = self._round_metric(self.curr_ep_q / self.curr_ep_loss_length)
        self.log_episode_metrics(self.curr_ep_reward, self.curr_ep_length, ep_avg_loss, ep_avg_q)
        self.init_episode()

    def log_episode_metrics(self, ep_reward, ep_length, ep_avg_loss=0, ep_avg_q=0):
        """记录一个已经统计完成的回合。"""
        self.ep_rewards.append(ep_reward)
        self.ep_lengths.append(ep_length)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def _round_metric(self, value):
        return round(float(value), METRIC_DECIMALS)

    def _round_epsilon(self, value):
        return round(float(value), EPSILON_DECIMALS)

    def _mean_last(self, values):
        if not values:
            return 0.0
        return self._round_metric(np.mean(values[-MEAN_WINDOW:]))

    def record(self, episode, epsilon, step):
        mean_ep_reward = self._mean_last(self.ep_rewards)
        mean_ep_length = self._mean_last(self.ep_lengths)
        mean_ep_loss = self._mean_last(self.ep_avg_losses)
        mean_ep_q = self._mean_last(self.ep_avg_qs)
        for moving_avg, mean in zip(
            self.moving_avgs,
            [mean_ep_reward, mean_ep_length, mean_ep_loss, mean_ep_q],
        ):
            moving_avg.append(mean)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = self._round_metric(self.record_time - last_record_time)

        tqdm.write(
            f"回合={episode} | "
            f"步数={step} | "
            f"探索率={self._round_epsilon(epsilon):.{EPSILON_DECIMALS}f} | "
            f"平均奖励={mean_ep_reward:.3f} | "
            f"平均长度={mean_ep_length:.3f} | "
            f"平均损失={mean_ep_loss:.3f} | "
            f"平均 Q 值={mean_ep_q:.3f} | "
            f"时间间隔={time_since_last_record:.3f}"
        )

        row = [
            episode,
            step,
            self._round_epsilon(epsilon),
            mean_ep_reward,
            mean_ep_length,
            mean_ep_loss,
            mean_ep_q,
            time_since_last_record,
        ]
        for csv_path in self.csv_paths:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

        for plot_paths in self.plot_path_groups:
            self._write_plots(plot_paths)

    def _write_plots(self, plot_paths):
        for moving_avg, plot_path, (title, xlabel, ylabel) in zip(
            self.moving_avgs,
            plot_paths,
            PLOT_CONFIGS,
        ):
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(11.69, 8.27))
            plt.plot(moving_avg, label=ylabel, linewidth=2)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
