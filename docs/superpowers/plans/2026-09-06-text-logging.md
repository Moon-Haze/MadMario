# 标准化 logging 重构（训练文本日志） Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Mad Mario 训练/播放 CLI 的散落打印统一为标准 `logging`，并让 `train` 命令每次启动把日志归档为按时间戳命名的人类可读文本文件，同时镜像 SB3 回调输出进同一文件。

**Architecture:** 新增 `mad_mario/logging/setup.py` 提供 `setup_logging()`（root logger 装配：stderr 终端 handler + 可选 FileHandler）与 `capture_stdout_to()`（上下文管理器，训练期把写往 stdout 的 SB3 文本镜像进日志文件）。项目自身消息走 logging（stderr + FileHandler），SB3 回调 `print` 走 stdout + 镜像——两者来源隔离、文件无重复。`cli.main()` 按命令装配：`train` 传 `save_root/logs` 并得到文件路径交给 `train(config, log_path=...)`；`play` 仅装配终端。

**Tech Stack:** Python ≥3.13 标准库 `logging` / `contextlib` / `datetime`；pytest（uv 环境）。

## Global Constraints

- 所有用户可见文本使用简体中文。
- 日志文件名精确匹配 `<prefix>_\d{8}_\d{6}.log`（如 `train_20260906_150422.log`）；格式串 `%(asctime)s [%(levelname)s] %(name)s: %(message)s`，asctime 日期格式 `%Y-%m-%d %H:%M:%S`。
- 终端 handler 指向 `sys.stderr`；文件 handler `mode="a"`、`encoding="utf-8"`。
- `setup_logging` 必须**可重入**：先移除并关闭 root 已有 handlers，再装配。
- 不改变 `latest_metrics.csv`、`latest_*.png`、`tensorboard/`、`snapshots/`、`best_model/`、`eval_logs/` 的路径与格式；不改变 SB3 各回调的 `verbose` 值。
- 磁盘/文件打开失败只降级为仅终端，不得让训练崩溃。
- 测试命令用 `uv run pytest <path> -v`（项目为 uv 工程，dev 组含 pytest）。
- 只 stage 任务清单列出的文件路径。**注意**：`src/mad_mario/config.py`、`src/mad_mario/cli.py`、`tests/test_config.py` 当前在主仓库有用户未提交改动，涉及这些文件时若在本工作区执行 `git add` 会连带用户的改动——建议在干净 worktree/分支上执行本计划，或与用户确认后再提交。

---

### Task 1: logging/setup.py 装配函数与文件名工具

**Files:**
- Create: `src/mad_mario/logging/setup.py`
- Test: `tests/test_logging.py`（新建）

**Interfaces:**
- Produces: `setup_logging(level: int = logging.INFO, log_dir: Path | None = None, prefix: str = "train", fmt: str = DEFAULT_FORMAT) -> Path | None`；常量 `DEFAULT_FORMAT`、`DATE_FORMAT`；`make_log_file_name(prefix: str, now: datetime | None = None) -> str`。供 Task 2、Task 6 使用。

- [ ] **Step 1: 写失败测试** `tests/test_logging.py`

```python
"""测试 logging/setup.py——装配函数、文件名与日志内容。"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import pytest

from mad_mario.logging.setup import make_log_file_name, setup_logging

LOG_NAME_RE = re.compile(r"train_\d{8}_\d{6}\.log")


@pytest.fixture(autouse=True)
def _isolate_root_logger():
    """隔离全局 root logger，避免测试互相污染。"""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_disabled = root.disabled
    yield
    root.disabled = saved_disabled
    root.setLevel(saved_level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for handler in saved_handlers:
        root.addHandler(handler)


class TestMakeLogFileName:
    def test_name(self):
        now = datetime(2026, 9, 6, 15, 4, 22)
        assert make_log_file_name("train", now) == "train_20260906_150422.log"

    def test_custom_prefix(self):
        now = datetime(2026, 1, 2, 3, 4, 5)
        assert make_log_file_name("eval", now) == "eval_20260102_030405.log"


class TestSetupLogging:
    def test_with_log_dir_writes_file(self, tmp_path: Path):
        log_path = setup_logging(logging.INFO, log_dir=tmp_path)
        assert log_path is not None
        assert LOG_NAME_RE.match(log_path.name)
        assert log_path.parent == tmp_path

        logging.getLogger("mad_mario.trainer").info("中文测试消息")
        assert log_path.exists()
        text = log_path.read_text(encoding="utf-8")
        assert "中文测试消息" in text
        assert "[INFO]" in text
        assert "mad_mario.trainer" in text

    def test_without_log_dir_returns_none(self):
        assert setup_logging(logging.INFO) is None

    def test_without_log_dir_has_terminal_handler(self):
        setup_logging(logging.INFO)
        root = logging.getLogger()
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    def test_level_filter(self, tmp_path: Path, capsys):
        setup_logging(logging.WARNING, log_dir=tmp_path)
        logging.getLogger("mad_mario").info("这条 INFO 不应落盘")
        logging.getLogger("mad_mario").warning("这条 WARNING 应落盘")
        files = list(tmp_path.glob("train_*.log"))
        assert files
        text = files[0].read_text(encoding="utf-8")
        assert "这条 WARNING 应落盘" in text
        assert "这条 INFO 不应落盘" not in text
        err = capsys.readouterr().err
        assert "这条 WARNING 应落盘" in err
        assert "这条 INFO 不应落盘" not in err
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_logging.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'mad_mario.logging.setup'`

- [ ] **Step 3: 最小实现** `src/mad_mario/logging/setup.py`

```python
"""标准 logging 装配：终端 + 按时间戳归档的训练文本日志。"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)


def make_log_file_name(prefix: str, now: datetime | None = None) -> str:
    """生成 ``<prefix>_YYYYMMDD_HHMMSS.log`` 风格的文件名。"""
    now = now or datetime.now()
    return f"{prefix}_{now:%Y%m%d_%H%M%S}.log"


def setup_logging(
    level: int = logging.INFO,
    log_dir: Path | None = None,
    prefix: str = "train",
    fmt: str = DEFAULT_FORMAT,
) -> Path | None:
    """装配 root logger：终端(stderr) + 可选按时间戳命名的文本文件。可重入。

    Args:
        level: 全局日志级别。
        log_dir: 非空时在此目录创建 ``<prefix>_YYYYMMDD_HHMMSS.log``。
        prefix: 日志文件名的前缀。
        fmt: 日志格式。

    Returns:
        日志文件路径；``log_dir`` 为 ``None`` 或文件创建失败时返回 ``None``。
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(level)
    formatter = logging.Formatter(fmt, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_dir is None:
        return None

    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / make_log_file_name(prefix)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        return log_path
    except OSError as exc:
        root.warning("无法创建日志文件，本次仅输出到终端: %s", exc)
        return None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_logging.py -v`
Expected: PASS（4 个 class，6 个用例）

- [ ] **Step 5: 提交**

```bash
git add src/mad_mario/logging/setup.py tests/test_logging.py
git commit -m "feat: 新增 logging/setup.py 日志装配（终端 + 时间戳文本文件）"
```

---

### Task 2: stdout 镜像 StdoutTee 与 capture_stdout_to

**Files:**
- Modify: `src/mad_mario/logging/setup.py`（追加镜像工具）
- Test: `tests/test_logging.py`（追加用例）

**Interfaces:**
- Consumes: Task 1 的 `setup_logging` 与模块 `logger`。
- Produces: `class StdoutTee`、`capture_stdout_to(log_path: Path | None)`（contextmanager，`None` 时为空操作）。供 Task 5（trainer）使用。

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_logging.py` 末尾，顶部 import 增加）

在文件顶部 import 增加 `import sys` 与 `from mad_mario.logging.setup import capture_stdout_to`：

```python
class TestCaptureStdoutTo:
    def test_mirrors_to_file_and_real_stdout(self, tmp_path: Path, capsys):
        log_path = tmp_path / "out.log"
        with capture_stdout_to(log_path):
            print("来自 stdout 的一行")
        out = capsys.readouterr().out
        assert "来自 stdout 的一行" in out
        assert log_path.read_text(encoding="utf-8") == "来自 stdout 的一行\n"

    def test_none_is_noop(self, tmp_path: Path):
        original = sys.stdout
        with capture_stdout_to(None):
            assert sys.stdout is original
            print("普通输出")
        assert sys.stdout is original
        assert list(tmp_path.glob("*.log")) == []

    def test_restores_stdout(self, tmp_path: Path):
        original = sys.stdout
        with capture_stdout_to(tmp_path / "out.log"):
            assert sys.stdout is not original
        assert sys.stdout is original
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_logging.py -v`
Expected: FAIL，`ImportError: cannot import name 'capture_stdout_to'`

- [ ] **Step 3: 追加实现到 `src/mad_mario/logging/setup.py`**（顶部 import 增加 `from contextlib import contextmanager` 与 `from typing import Iterator`；文件末尾追加）

```python
class StdoutTee:
    """把写入同时转发到真实 stdout 与一个镜像日志文件，写入后即时 flush。"""

    def __init__(self, real_stdout, log_path: Path):
        self._real = real_stdout
        self._file = open(log_path, mode="a", encoding="utf-8")
        self._file_ok = True

    def write(self, text: str) -> int:
        self._real.write(text)
        try:
            self._real.flush()
        except OSError:
            pass
        if self._file_ok:
            try:
                self._file.write(text)
                self._file.flush()
            except OSError:
                self._file_ok = False
                self._real.write("\n[logging] 镜像日志文件写入失败，已停止镜像到文件。\n")
        return len(text)

    def flush(self) -> None:
        self._real.flush()
        if self._file_ok:
            self._file.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._file.close()

    def isatty(self) -> bool:
        return self._real.isatty()

    def fileno(self) -> int:
        return self._real.fileno()

    def __getattr__(self, name: str):
        # 其余属性（encoding/buffer 等）委托给真实 stdout
        return getattr(self._real, name)


@contextmanager
def capture_stdout_to(log_path: Path | None) -> Iterator[None]:
    """把写往 stdout 的文本同时镜像到日志文件；``log_path`` 为 ``None`` 时为空操作。"""
    if log_path is None:
        yield
        return
    real_stdout = sys.stdout
    try:
        tee = StdoutTee(real_stdout, log_path)
    except OSError as exc:
        logger.warning("无法打开 stdout 镜像日志文件，本次仅终端输出: %s", exc)
        yield
        return
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = real_stdout
        tee.close()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_logging.py -v`
Expected: PASS（新增 3 个用例全绿）

- [ ] **Step 5: 提交**

```bash
git add src/mad_mario/logging/setup.py tests/test_logging.py
git commit -m "feat: 添加 stdout 镜像 capture_stdout_to 以归档 SB3 文本"
```

---

### Task 3: config 新增 --log-level

**Files:**
- Modify: `src/mad_mario/config.py`（SB3ArtifactConfig 字段、add_train_args、config_from_train_args）
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `SB3ArtifactConfig.log_level: str = "info"`；train 子命令 `--log-level`（choices=`debug/info/warning/error`）；`config_from_train_args` 返回的 config 携带 `artifacts.log_level`。供 Task 6 的 cli 读取 `args.log_level`。

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_config.py` 的 `TestConfigFromTrainArgs` 类内）

```python
    def test_log_level_default(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args([])
        config = config_from_train_args(args)
        assert config.artifacts.log_level == "info"

    def test_log_level_custom(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(["--log-level", "debug"])
        config = config_from_train_args(args)
        assert config.artifacts.log_level == "debug"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_config.py -k log_level -v`
Expected: FAIL（`AttributeError: ... has no attribute 'log_level'`）

- [ ] **Step 3: 实现**

(a) `SB3ArtifactConfig` 字段（`run_name` 之后加一行，注意缩进与注释对齐）：

```python
    run_name: str | None = None
    log_level: str = "info"            # logging 级别：debug/info/warning/error
```

(b) `add_train_args` 的 `art` 组末尾（`--log-interval` 之后）追加：

```python
    art.add_argument("--log-level", choices=("debug", "info", "warning", "error"),
                     default=SB3ArtifactConfig.log_level,
                     help="日志级别（默认 info）")
```

(c) `config_from_train_args` 中 `artifact_config = SB3ArtifactConfig(...)` 增加一项：

```python
        log_interval=args.log_interval,
        log_level=args.log_level,
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS（含既有用例）

- [ ] **Step 5: 提交**

```bash
git add src/mad_mario/config.py tests/test_config.py
git commit -m "feat: 新增 --log-level 训练日志级别配置项"
```

---

### Task 4: metrics.py 指标行改用 logging

**Files:**
- Modify: `src/mad_mario/logging/metrics.py`（import、`MarioMetricsCallback._record`、`MetricLogger.record`）
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: 无（仅标准库 logging）。
- Produces: 两个记录函数均通过 `logger = logging.getLogger(__name__)`（名 `mad_mario.logging.metrics`）输出 INFO 文本，不再使用 `tqdm.write`。

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_metrics.py`；顶部 import 增加 `import logging` 与 `from types import SimpleNamespace`）

```python
class TestRecordLogging:
    def test_record_emits_text_via_logging(self, tmp_path: Path, caplog):
        callback = MarioMetricsCallback(save_dir=tmp_path, log_interval=20)
        callback.csv_path = tmp_path / "latest_metrics.csv"
        callback._ensure_csv_header()
        callback._write_plots = lambda: None  # 避免真实绘图
        callback.model = SimpleNamespace(exploration_rate=0.5)
        callback.num_timesteps = 1000
        callback.episode_count = 20
        callback.ep_rewards = [10.0, 20.0]
        with caplog.at_level(logging.INFO, logger="mad_mario.logging.metrics"):
            callback._record()
        assert "回合=20" in caplog.text
        assert "平均奖励=" in caplog.text
        assert "平均最大 x=" in caplog.text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_metrics.py::TestRecordLogging -v`
Expected: FAIL（`caplog.text` 为空——当前文本走 tqdm.write 到 stdout，不进 logging）

- [ ] **Step 3: 实现**

(a) import 区调整——删除 `from tqdm import tqdm`，新增 `import logging`，并在第三方 import 之后定义模块 logger：

```python
import csv
import logging
import time
from collections import deque
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)
```

(b) `MarioMetricsCallback._record()` 中 `tqdm.write(...)` 整段替换为：

```python
        logger.info(
            "回合=%s | 步数=%s | 探索率=%.4f | 平均奖励=%.3f | 平均长度=%.3f | "
            "平均损失=%.3f | 平均 Q 值=%.3f | 平均最大 x=%.3f | 时间间隔=%.3f",
            self.episode_count, self.num_timesteps, eps,
            mean_rew, mean_len, mean_loss, mean_q, mean_max_x, delta,
        )
```

(c) 旧版 `MetricLogger.record(..., emit=True)` 内的 `tqdm.write(...)` 整段替换为：

```python
        logger.info(
            "回合=%s | 步数=%s | 探索率=%.4f | 平均奖励=%.3f | 平均长度=%.3f | "
            "平均损失=%.3f | 平均 Q 值=%.3f | 平均最大 x=%.3f | "
            "平均 NOOP=%.3f | 常用动作=%d | 时间间隔=%.3f",
            episode, step, epsilon,
            mean_ep_reward, mean_ep_length, mean_ep_loss, mean_ep_q,
            mean_max_x_pos, mean_noop_rate, int(mean_most_used_action), delta,
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS（新增用例 + 既有 `TestMeanLast`/`TestCsvHeader` 全绿）

- [ ] **Step 5: 提交**

```bash
git add src/mad_mario/logging/metrics.py tests/test_metrics.py
git commit -m "refactor: 指标记录改用标准 logging 输出"
```

---

### Task 5: trainer 改用 logging 并接入 stdout 镜像

**Files:**
- Modify: `src/mad_mario/training/trainer.py`
- Test: `tests/test_logging.py`（追加迁移守卫用例）

**Interfaces:**
- Consumes: Task 2 的 `capture_stdout_to`；config 结构不变。
- Produces: `train(config: AppConfig, log_path: Path | None = None) -> None`（Task 6 的 cli 以 `train(config, log_path=log_path)` 调用）。

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_logging.py` 末尾；顶部 import 增加 `from pathlib import Path`——已在 Task 1 引入）

```python
class TestTrainerMigration:
    def test_trainer_off_tqdm_and_wired_for_capture(self):
        repo = Path(__file__).resolve().parents[1]
        src = (repo / "src" / "mad_mario" / "training" / "trainer.py").read_text(encoding="utf-8")
        assert "tqdm" not in src
        assert "capture_stdout_to" in src
        assert "log_path: Path | None = None" in src
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_logging.py::TestTrainerMigration -v`
Expected: FAIL（`assert "tqdm" not in src`——当前文件含 `from tqdm import tqdm`）

- [ ] **Step 3: 实现 `trainer.py`**

(a) stdlib import 增加 `import logging`（放在 `from __future__ import annotations` 之后、`import random` 之前或同组均可，保持字母序风格）：

```python
import logging
import random
from pathlib import Path
```

(b) 删除 `from tqdm import tqdm` 这一行；新增 mad_mario 导入与模块 logger（放在 `from mad_mario.logging.metrics import MarioMetricsCallback` 之后）：

```python
from mad_mario.logging.metrics import MarioMetricsCallback
from mad_mario.logging.setup import capture_stdout_to

logger = logging.getLogger(__name__)
```

(c) 将整个 `train` 函数体替换为下面版本（**签名新增 `log_path`；所有 `tqdm.write` → `logger.info/warning`；learn+保存包进 `capture_stdout_to`**）：

```python
def train(config: AppConfig, log_path: Path | None = None) -> None:
    """使用 Stable-Baselines3 DQN 训练 Mario 智能体。

    - 创建或加载 DQN 模型
    - 每 ``save_freq`` 步自动保存 checkpoint（含经验回放缓冲区）
    - 每 ``log_interval`` 回合输出日志并刷新图表
    - 每 ``eval_freq`` 步执行一次评估
    - 支持 ``Ctrl+C`` 安全中断后恢复

    Args:
        config: 顶层配置。
        log_path: 非空时把训练期间写往 stdout 的文本（SB3 回调输出）
            镜像到该日志文件；默认 ``None``（不镜像）。
    """
    save_root = config.artifacts.save_root
    save_root.mkdir(parents=True, exist_ok=True)

    # ── 1. 创建并行环境 ──
    logger.info("创建 %s 个并行环境...", config.artifacts.num_envs)
    env_fns = _make_env_thunks(config)
    env = SubprocVecEnv(env_fns)

    # ── 2. DQN 模型参数 ──
    policy_kwargs = config.sb3.policy_kwargs or {}
    dqn_params: dict[str, Any] = dict(
        policy="CnnPolicy",
        env=env,
        learning_rate=config.sb3.learning_rate,
        buffer_size=config.sb3.buffer_size,
        learning_starts=config.sb3.learning_starts,
        batch_size=config.sb3.batch_size,
        tau=config.sb3.tau,
        gamma=config.sb3.gamma,
        train_freq=config.sb3.train_freq,
        gradient_steps=config.sb3.gradient_steps,
        target_update_interval=config.sb3.target_update_interval,
        exploration_fraction=config.sb3.exploration_fraction,
        exploration_initial_eps=config.sb3.exploration_initial_eps,
        exploration_final_eps=config.sb3.exploration_final_eps,
        max_grad_norm=config.sb3.max_grad_norm,
        seed=config.sb3.seed,
        device="auto",
        policy_kwargs=policy_kwargs,
        verbose=0,
        stats_window_size=100,
        tensorboard_log=str(save_root / "tensorboard"),
    )

    # ── 3. 创建或加载模型 ──
    model: DQN | None = None
    loaded = False

    if config.artifacts.resume:
        ckpt_path = resolve_checkpoint(config.artifacts)
        if ckpt_path is not None and ckpt_path.exists():
            try:
                logger.info("正在恢复 checkpoint: %s", ckpt_path)
                model = DQN.load(ckpt_path, env=env, device="auto")
                loaded = True
            except Exception as exc:
                logger.warning("checkpoint 加载失败，将从头开始训练。原因: %s", exc)

    if model is None:
        logger.info("从头开始创建 DQN 模型...")
        model = DQN(**dqn_params)

    # ── 4. 回调 ──
    save_freq = max(config.artifacts.save_freq // config.artifacts.num_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(save_root / "snapshots"),
        name_prefix="mario_dqn",
        save_replay_buffer=config.artifacts.save_replay_buffer,
        save_vecnormalize=False,
        verbose=1,
    )

    # 评估回调（SB3 内置，自动保存 best model + 记录评估日志）
    eval_env = SubprocVecEnv([lambda: Monitor(_env_thunk(config.env)())])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_root / "best_model"),
        log_path=str(save_root / "eval_logs"),
        eval_freq=max(config.artifacts.eval_freq // config.artifacts.num_envs, 1),
        n_eval_episodes=config.artifacts.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    # 自定义指标回调（CSV + PNG 图表 + Mario 特有指标）
    metrics_callback = MarioMetricsCallback(
        save_dir=save_root,
        log_interval=config.artifacts.log_interval,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback, metrics_callback])

    # ── 5. 训练（镜像 SB3 文本到 log_path）──
    logger.info(
        "开始训练（总步数=%s, 并行数=%s, 保存间隔=%s 步）",
        config.sb3.total_timesteps,
        config.artifacts.num_envs,
        config.artifacts.save_freq,
    )
    try:
        with capture_stdout_to(log_path):
            try:
                model.learn(
                    total_timesteps=config.sb3.total_timesteps,
                    callback=callbacks,
                    reset_num_timesteps=not loaded,
                    log_interval=1,  # SB3 内部日志频率（每 N 步），和我们的回调独立
                )
            except KeyboardInterrupt:
                logger.info("检测到键盘中断，正在保存最终 checkpoint...")

            # ── 6. 保存 ──
            final_path = str(save_root / "mario_dqn_final.zip")
            model.save(final_path)
            logger.info("最终模型已保存到 %s", final_path)
    finally:
        env.close()
        logger.info("训练完成。")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_logging.py tests/test_metrics.py -v`
Expected: PASS（迁移守卫 + 全量相关用例）

- [ ] **Step 5: 提交**

```bash
git add src/mad_mario/training/trainer.py tests/test_logging.py
git commit -m "refactor: trainer 改用 logging 并接入 stdout 镜像归档"
```

---

### Task 6: cli.py 装配 logging 并统一 play 输出

**Files:**
- Modify: `src/mad_mario/cli.py`
- Test: `tests/test_cli_logging.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 `setup_logging`；Task 3 的 `args.log_level`；Task 5 的 `train(config, log_path=...)`。
- Produces: `main()` 在 `train` 分支装配 `setup_logging(level=..., log_dir=config.artifacts.save_root / "logs")` 并把返回路径传给 `train`；`play` 分支仅 `setup_logging()`。`play()` 全部 `print` → `logger`。

- [ ] **Step 1: 写失败测试** `tests/test_cli_logging.py`

```python
"""测试 cli.py 的日志装配接线（mock 掉重依赖，验证调用关系）。"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest


@pytest.fixture
def fake_setup_logging_factory():
    def _make(calls):
        def fake_setup_logging(level=logging.INFO, log_dir=None, prefix="train"):
            calls["level"] = level
            calls["log_dir"] = log_dir
            calls["prefix"] = prefix
            return None
        return fake_setup_logging
    return _make


class TestMainTrainWiring:
    def test_configures_file_logging_and_passes_log_path(self, tmp_path, monkeypatch):
        import mad_mario.cli as cli_mod

        fake_save_root = tmp_path / "saves"
        fake_config = SimpleNamespace(artifacts=SimpleNamespace(save_root=fake_save_root))
        fake_log_path = tmp_path / "logs" / "train_20260906_150422.log"

        monkeypatch.setattr(cli_mod, "config_from_train_args", lambda args: fake_config)
        calls: dict = {}
        monkeypatch.setattr(cli_mod, "setup_logging", fake_setup_logging_factory()(calls))
        train_calls: dict = {}

        def fake_train(config, log_path=None):
            train_calls["config"] = config
            train_calls["log_path"] = log_path

        monkeypatch.setattr(cli_mod, "train", fake_train)
        monkeypatch.setattr("sys.argv", ["mad-mario", "train", "--log-level", "debug"])

        cli_mod.main()

        assert calls["log_dir"] == fake_save_root / "logs"
        assert calls["level"] == logging.DEBUG
        assert train_calls["config"] is fake_config
        assert train_calls["log_path"] is None  # fake_setup_logging 返回 None


class TestMainPlayWiring:
    def test_terminal_only_and_reaches_play(self, tmp_path, monkeypatch):
        import mad_mario.cli as cli_mod

        fake_config = SimpleNamespace()
        monkeypatch.setattr(cli_mod, "config_from_play_args", lambda args: fake_config)
        calls: dict = {}
        monkeypatch.setattr(cli_mod, "setup_logging", fake_setup_logging_factory()(calls))
        play_calls: dict = {}

        def fake_play(config):
            play_calls["config"] = config

        monkeypatch.setattr(cli_mod, "play", fake_play)
        monkeypatch.setattr("sys.argv", ["mad-mario", "play"])

        cli_mod.main()

        assert calls["log_dir"] is None
        assert calls["level"] == logging.INFO
        assert play_calls["config"] is fake_config
```

> 说明：两个测试在函数体内 `import mad_mario.cli`（首次导入会拉入 SB3 DQN/工厂，耗时数秒属正常）。因所有重函数都被 mock，不会真正创建环境或模型。

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_cli_logging.py -v`
Expected: FAIL（`main()` 仍直接 `train(config_from_train_args(args))`，未调用 `setup_logging`，`calls` 为空）

- [ ] **Step 3: 实现 `cli.py`**

(a) 顶部 stdlib import 组增加 `import logging`：

```python
import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
```

(b) mad_mario import 区新增 `setup_logging` 并定义模块 logger（放在 `from mad_mario.training.trainer import train` 之后）：

```python
from mad_mario.env.factory import make_mario_env
from mad_mario.logging.setup import setup_logging
from mad_mario.training.trainer import train

logger = logging.getLogger(__name__)
```

(c) `main()` 末尾分支替换为：

```python
    if args.command == "train":
        config = config_from_train_args(args)
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        log_path = setup_logging(level=log_level, log_dir=config.artifacts.save_root / "logs")
        train(config, log_path=log_path)
    elif args.command == "play":
        setup_logging()
        play(config_from_play_args(args))
    else:
        parser.error(f"未知命令: {args.command}")
```

(d) `play()` 内全部 `print` 替换（对照下表逐处替换，均为同一条调用上下文内改动，其余代码不动）：

```python
        logger.error(f"错误：checkpoint 不存在（{checkpoint_path}）")
```
```python
            logger.info(f"正在加载 checkpoint: {checkpoint_path}")
            logger.info(f"模型已加载。已训练步数: {model.num_timesteps}")
```
```python
        logger.info("=== 播放关卡 %s（%d/%d）===", level_label, level_idx, len(levels))
```
```python
                logger.info(
                    "关卡 %s：步数=%d，奖励=%.1f，通关=%s",
                    level_label, step, total_reward, flag_get,
                )
```

> `play()` 中原 `print(f"\n=== ...")` 前导的换行不再保留（logging 自带换行）；`logger.error` 分支的 `return` 逻辑保持不变。

- [ ] **Step 4: 运行测试 + 冒烟确认通过**

Run: `uv run pytest tests/test_cli_logging.py tests/test_logging.py -v`
Expected: PASS

Run（CLI 冒烟，确认导入、子命令与 `--log-level` 接线无误、无环境启动）：
```bash
uv run python -m mad_mario.cli train --help
uv run python -m mad_mario.cli play --help
```
Expected: 两处 help 正常打印，train help 中出现 `--log-level`。

- [ ] **Step 5: 全量回归 + 提交**

Run: `uv run pytest -q`
Expected: 全绿

```bash
git add src/mad_mario/cli.py tests/test_cli_logging.py
git commit -m "refactor: cli 装配 logging 并统一 play 文本输出"
```

---

## 端到端人工验证（可选，须有 SuperMario 环境）

真正的端到端（含 SB3 回调文本镜像）需要真实环境，无法在 pytest 覆盖。可在本机手动执行：

```bash
uv run python -m mad_mario.cli train --timesteps 2000 --learning-starts 100 --log-interval 2 --save-root /tmp/mario_logging_smoke
```

预期：终端出现带 `[INFO]` 与时间戳的进度行；`/tmp/mario_logging_smoke/1-1-v0_right_only/logs/` 下生成 `train_<时间戳>.log`；文件同时包含我们 logging 的行与 SB3 评估/保存文本。确认后可 `rm -rf /tmp/mario_logging_smoke`。
