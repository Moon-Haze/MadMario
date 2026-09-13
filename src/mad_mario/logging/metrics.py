import csv
import logging
import time
from collections import deque
from pathlib import Path

import matplotlib
# 本模块只把曲线写成 PNG，从不开窗口，需要非交互后端。后端必须由入口在 pyplot
# 首次导入之前定好（cli._use_headless_backend），到这里再改已经晚了——TkAgg
# 下主线程建出的 Tk 对象，会被训练里那个录制最优模型的后台线程的 GC 回收
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


METRIC_DECIMALS = 3
EPSILON_DECIMALS = 4
MEAN_WINDOW = 100
PLOT_CONFIGS = [
    ("平均奖励曲线", "回合", "平均奖励", "reward"),
    ("平均长度曲线", "回合", "平均长度", "length"),
    ("平均损失曲线", "回合", "平均损失", "loss"),
    ("平均 Q 值曲线", "回合", "平均 Q 值", "q"),
]

# 断点重启时回读历史所用的 CSV 列，顺序必须与 PLOT_CONFIGS 一一对应
HISTORY_COLUMNS = ["mean_reward", "mean_length", "mean_loss", "mean_q"]

# 动作预设中 0 号是 NOOP（gym_super_mario_bros.actions 的 RIGHT_ONLY/SIMPLE 皆然）
NOOP_ACTION = 0

# 估计平均 Q 值时从回放池采样的状态数
Q_SAMPLE_SIZE = 256

# 单张图最多渲染的点数。断点接续后历史点会不断累积，等距抽样可在
# 肉眼无差别的前提下，避免长训练后期每次画图都遍历全部历史点。
MAX_PLOT_POINTS = 2000


def _downsample(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """把一条曲线等距抽稀到 ``MAX_PLOT_POINTS`` 点以内，返回 (x, y)。

    x 轴仍是该点在完整历史中的真实序号（只是被抽稀），因此四张图的
    横轴含义保持一致。
    """
    count = len(values)
    if count <= MAX_PLOT_POINTS:
        return np.arange(1, count + 1), np.asarray(values, dtype=float)
    indices = np.unique(np.linspace(0, count - 1, MAX_PLOT_POINTS).astype(int))
    return indices + 1, np.asarray(values, dtype=float)[indices]


# Matplotlib 中文字体配置
plt.rcParams["font.sans-serif"] = [
    "Source Han Sans SC", "Noto Sans CJK SC", "LXGW WenKai Mono", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


# ──────────────────────────────────────────────
# SB3 训练回调
# ──────────────────────────────────────────────
class MarioMetricsCallback(BaseCallback):
    """在 SB3 训练过程中收集指标、写入 CSV、生成 PNG 图表。

    - 从 ``infos`` 中追踪回合奖励 / 长度 / 最大 x 位置
    - 从 SB3 logger 中读取 loss / learning_rate
    - 每 ``log_interval`` 回合写一行 CSV、刷新 PNG
    - 评估由 SB3 内置的 ``EvalCallback`` 完成
    """

    def __init__(
        self,
        save_dir: Path,
        log_interval: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval

        # 每个 episode 的原始指标
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int] = []
        self.ep_avg_losses: list[float] = []
        self.ep_avg_qs: list[float] = []
        self.ep_max_x_positions: list[float] = []
        self.ep_noop_rates: list[float] = []
        self.ep_most_used_actions: list[int] = []

        # 移动平均历史（每次 record 追加一个点）
        self.moving_avgs: list[list[float]] = [[], [], [], []]

        # 当前 episode 内累积的 loss (用于计算 episode 平均 loss)
        self._loss_buffer: deque[float] = deque(maxlen=256)

        self.episode_count = 0
        self.record_time = time.time()

    # ── BaseCallback 接口 ──

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._rew = np.zeros(n_envs, dtype=np.float32)
        self._len = np.zeros(n_envs, dtype=np.int32)
        self._max_x = np.zeros(n_envs, dtype=np.float32)
        self._noop = np.zeros(n_envs, dtype=np.int32)
        self._act_count = [np.zeros(self.model.action_space.n, dtype=np.int32) for _ in range(n_envs)]

        # 防止同一个 episode_count 重复触发 record
        self._last_recorded_episode = 0

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.save_dir / "latest_metrics.csv"
        self._ensure_csv_header()
        # 断点重启时把已有曲线的历史读回来，图表接着上一次继续画
        self._load_history_from_csv()

    def _on_step(self) -> bool:
        """每个环境步调用一次。"""
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        # 本步真正下发给各环境的动作。环境自身不会往 info 里写动作，只能从
        # SB3 的 locals 取；取不到时动作统计会全部退化成 0 号。
        actions = self.locals.get("actions", ())

        # 收集 loss (SB3 的 DQN 在 train 阶段写入 logger)
        train_loss = self.logger.name_to_value.get("train/loss", None)
        if train_loss is not None:
            self._loss_buffer.append(float(train_loss))

        # 同一步内可能有多个环境同时结束回合，Q 估计只做一次
        q_estimate = None

        for i, info in enumerate(infos):
            self._rew[i] += float(rewards[i]) if i < len(rewards) else 0.0
            self._len[i] += 1
            self._max_x[i] = max(self._max_x[i], float(info.get("x_pos", 0.0)))

            # 动作计数
            action = int(actions[i]) if i < len(actions) else NOOP_ACTION
            if i < len(self._act_count) and 0 <= action < len(self._act_count[i]):
                self._act_count[i][action] += 1
                if action == NOOP_ACTION:
                    self._noop[i] += 1

            # episode 结束时记录
            if dones[i]:
                ep_rew = float(self._rew[i])
                ep_len = int(self._len[i])
                ep_max_x = float(self._max_x[i])
                noop_rate = float(self._noop[i]) / max(1, ep_len)
                most_used = int(np.argmax(self._act_count[i]))
                ep_loss = float(np.mean(self._loss_buffer)) if self._loss_buffer else 0.0
                if q_estimate is None:
                    q_estimate = self._estimate_q()

                self.ep_rewards.append(ep_rew)
                self.ep_lengths.append(ep_len)
                self.ep_avg_losses.append(round(ep_loss, METRIC_DECIMALS))
                self.ep_avg_qs.append(q_estimate)
                self.ep_max_x_positions.append(round(ep_max_x, METRIC_DECIMALS))
                self.ep_noop_rates.append(round(noop_rate, METRIC_DECIMALS))
                self.ep_most_used_actions.append(most_used)
                self.episode_count += 1

                # 重置累积器
                self._rew[i] = 0.0
                self._len[i] = 0
                self._max_x[i] = 0.0
                self._noop[i] = 0
                self._act_count[i].fill(0)

        # 定频记录（每个 episode_count 只触发一次）
        if self.episode_count > 0 and self.episode_count % self.log_interval == 0 \
                and self.episode_count > self._last_recorded_episode:
            self._last_recorded_episode = self.episode_count
            self._record()

        return True

    def _estimate_q(self) -> float:
        """从回放池采样一批状态，估计当前 Q 网络的平均最大 Q 值。

        取 ``max`` 是因为 DQN 依靠最大 Q 选择动作，该值直接反映策略对当前
        状态价值的判断。回放池积累的样本不足采样数时返回 0.0。
        """
        replay_buffer = getattr(self.model, "replay_buffer", None)
        if replay_buffer is None or replay_buffer.size() < Q_SAMPLE_SIZE:
            return 0.0
        with torch.no_grad():
            # sample() 返回的观测已由 SB3 搬到模型所在 device
            observations = replay_buffer.sample(Q_SAMPLE_SIZE).observations
            # q_net 是 DQN 特有属性，基类 BaseAlgorithm 上没有
            q_values = self.model.q_net(observations)  # type: ignore[attr-defined]
        return round(float(q_values.max(dim=1).values.mean().item()), METRIC_DECIMALS)

    # ── 记录与绘图 ──

    def _record(self) -> None:
        """将最近 MEAN_WINDOW 个 episode 的均值写入 CSV 并刷新图表。"""
        mean_rew = self._mean_last(self.ep_rewards)
        mean_len = self._mean_last(self.ep_lengths)
        mean_loss = self._mean_last(self.ep_avg_losses)
        mean_q = self._mean_last(self.ep_avg_qs)
        mean_max_x = self._mean_last(self.ep_max_x_positions)

        self.moving_avgs[0].append(mean_rew)
        self.moving_avgs[1].append(mean_len)
        self.moving_avgs[2].append(mean_loss)
        self.moving_avgs[3].append(mean_q)

        now = time.time()
        delta = round(now - self.record_time, METRIC_DECIMALS)
        self.record_time = now

        eps = self.model.exploration_rate if hasattr(self.model, "exploration_rate") else 0.0

        row = [
            self.episode_count,
            self.num_timesteps,
            round(eps, EPSILON_DECIMALS),
            mean_rew, mean_len, mean_loss, mean_q,
            mean_max_x,
            self._mean_last(self.ep_noop_rates),
            self._mean_last(self.ep_most_used_actions, as_int=True),
            delta,
        ]
        self._append_csv(row)

        logger.info(
            "回合=%s | 步数=%s | 探索率=%.4f | 平均奖励=%.3f | 平均长度=%.3f | "
            "平均损失=%.3f | 平均 Q 值=%.3f | 平均最大 x=%.3f | 时间间隔=%.3f",
            self.episode_count, self.num_timesteps, eps,
            mean_rew, mean_len, mean_loss, mean_q, mean_max_x, delta,
        )

        self._write_plots()

    # ── CSV 操作 ──

    def _ensure_csv_header(self):
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "step", "epsilon",
                "mean_reward", "mean_length", "mean_loss", "mean_q",
                "mean_max_x_pos", "mean_noop_rate", "mean_most_used_action",
                "time_delta",
            ])

    def _append_csv(self, row: list):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def _load_history_from_csv(self) -> None:
        """把 CSV 中已有的曲线历史读回内存，使图表跨断点接续绘制。

        新进程的 ``moving_avgs`` 是空的；若不回读，图表会从第 1 个记录点
        重新画起，丢掉此前全部曲线。这里以 CSV 为唯一事实来源补齐历史，
        因此中断、重启、续训都能接着上一次继续画。

        列缺失或无法解析的行直接跳过——训练被强杀时最后一行可能只写了
        一半。
        """
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    values = [float(row[column]) for column in HISTORY_COLUMNS]
                except (KeyError, TypeError, ValueError):
                    continue
                for series, value in zip(self.moving_avgs, values):
                    series.append(value)

    # ── 绘图 ──

    def _write_plots(self):
        for moving_avg, (title, xlabel, ylabel, fname) in zip(self.moving_avgs, PLOT_CONFIGS):
            plot_path = self.save_dir / f"latest_{fname}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            episodes, values = _downsample(moving_avg)
            plt.figure(figsize=(11.69, 8.27))
            plt.plot(episodes, values, label=ylabel, linewidth=2)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

    # ── 工具 ──

    def _mean_last(self, values, as_int=False):
        if not values:
            return 0
        mean_val = float(np.mean(values[-MEAN_WINDOW:]))
        return int(round(mean_val)) if as_int else round(mean_val, METRIC_DECIMALS)


# ──────────────────────────────────────────────
# 旧版 MetricLogger（保留以供参考，不再主动使用）
# ──────────────────────────────────────────────
class MetricLogger:
    """保持向后兼容的旧版日志记录器。新代码请使用 ``MarioMetricsCallback``。"""

    def __init__(self, csv_paths: list[Path], plot_path_groups: list[list[Path]], reset: bool = False):
        self.csv_paths = csv_paths
        self.plot_path_groups = plot_path_groups
        if reset:
            self._clear_outputs()
        for csv_path in self.csv_paths:
            self._ensure_csv_header(csv_path)

        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int] = []
        self.ep_avg_losses: list[float] = []
        self.ep_avg_qs: list[float] = []
        self.ep_max_x_positions: list[float] = []
        self.ep_noop_rates: list[float] = []
        self.ep_most_used_actions: list[int] = []
        self.moving_avgs: list[list[float]] = [[], [], [], []]
        self.max_x_pos_avgs: list[float] = []
        self.noop_rate_avgs: list[float] = []
        self.most_used_action_avgs: list[float] = []
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
                "episode", "step", "epsilon",
                "mean_reward", "mean_length", "mean_loss", "mean_q",
                "mean_max_x_pos", "mean_noop_rate", "mean_most_used_action",
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
                    self.max_x_pos_avgs.append(float(row.get("mean_max_x_pos", 0.0)))
                    self.noop_rate_avgs.append(float(row.get("mean_noop_rate", 0.0)))
                    self.most_used_action_avgs.append(float(row.get("mean_most_used_action", 0.0)))
                except (KeyError, TypeError, ValueError):
                    continue

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self, max_x_pos=0.0, noop_rate=0.0, most_used_action=0):
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = round(self.curr_ep_loss / self.curr_ep_loss_length, METRIC_DECIMALS)
            ep_avg_q = round(self.curr_ep_q / self.curr_ep_loss_length, METRIC_DECIMALS)
        self.log_episode_metrics(
            self.curr_ep_reward, self.curr_ep_length,
            ep_avg_loss, ep_avg_q, max_x_pos, noop_rate, most_used_action,
        )
        self.init_episode()

    def log_episode_metrics(self, ep_reward, ep_length, ep_avg_loss=0, ep_avg_q=0,
                            max_x_pos=0.0, noop_rate=0.0, most_used_action=0):
        self.ep_rewards.append(ep_reward)
        self.ep_lengths.append(ep_length)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.ep_max_x_positions.append(float(max_x_pos))
        self.ep_noop_rates.append(float(noop_rate))
        self.ep_most_used_actions.append(int(most_used_action))

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def _mean_last(self, values):
        if not values:
            return 0.0
        return round(float(np.mean(values[-MEAN_WINDOW:])), METRIC_DECIMALS)

    def record(self, episode, epsilon, step, emit=False):
        mean_ep_reward = self._mean_last(self.ep_rewards)
        mean_ep_length = self._mean_last(self.ep_lengths)
        mean_ep_loss = self._mean_last(self.ep_avg_losses)
        mean_ep_q = self._mean_last(self.ep_avg_qs)
        mean_max_x_pos = self._mean_last(self.ep_max_x_positions)
        mean_noop_rate = self._mean_last(self.ep_noop_rates)
        mean_most_used_action = self._mean_last(self.ep_most_used_actions)

        self.moving_avgs[0].append(mean_ep_reward)
        self.moving_avgs[1].append(mean_ep_length)
        self.moving_avgs[2].append(mean_ep_loss)
        self.moving_avgs[3].append(mean_ep_q)
        self.max_x_pos_avgs.append(mean_max_x_pos)
        self.noop_rate_avgs.append(mean_noop_rate)
        self.most_used_action_avgs.append(mean_most_used_action)

        now = time.time()
        delta = round(now - self.record_time, METRIC_DECIMALS)
        self.record_time = now

        row = [
            episode, step, round(epsilon, EPSILON_DECIMALS),
            mean_ep_reward, mean_ep_length, mean_ep_loss, mean_ep_q,
            mean_max_x_pos, mean_noop_rate, round(mean_most_used_action, 0),
            delta,
        ]
        for csv_path in self.csv_paths:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

        if not emit:
            return

        logger.info(
            "回合=%s | 步数=%s | 探索率=%.4f | 平均奖励=%.3f | 平均长度=%.3f | "
            "平均损失=%.3f | 平均 Q 值=%.3f | 平均最大 x=%.3f | "
            "平均 NOOP=%.3f | 常用动作=%d | 时间间隔=%.3f",
            episode, step, epsilon,
            mean_ep_reward, mean_ep_length, mean_ep_loss, mean_ep_q,
            mean_max_x_pos, mean_noop_rate, int(mean_most_used_action), delta,
        )

        for plot_paths in self.plot_path_groups:
            self._write_plots(plot_paths)

    def _write_plots(self, plot_paths=None):
        paths = plot_paths or [self.plot_path_groups[0]] if self.plot_path_groups else []
        for plot_group in ([paths] if not isinstance(paths[0], list) else paths):
            for moving_avg, plot_path, (title, xlabel, ylabel, fname) in zip(
                self.moving_avgs, plot_group, PLOT_CONFIGS,
            ):
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(11.69, 8.27))
                episodes = range(1, len(moving_avg) + 1)
                plt.plot(episodes, moving_avg, linewidth=2)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.legend([ylabel])
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
