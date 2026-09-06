# 标准化 logging 重构：训练文本日志输出

- 日期：2026-09-06
- 状态：已评审（brainstorming 两节设计均获确认）
- 范围：Mad Mario DQN 训练 / 播放 CLI

## 1. 背景与目标

当前训练过程的文本输出通过散落的 `tqdm.write`（trainer.py、metrics.py）与 `print`（cli.py）打到终端，唯一的文件文本是结构化 CSV（`latest_metrics.csv`）。存在两个问题：

1. 输出方式不统一，无日志级别概念，难以分级过滤。
2. 终端文本无法留痕——后台 / 长时训练结束后无法回看人类可读的日志。

**目标**：把项目自身全部打印输出统一为 Python 标准 `logging`；`train` 命令自动把日志归档为按时间戳命名的人类可读文本文件；SB3 回调（`EvalCallback`/`CheckpointCallback`）的 `print` 文本尽量一并归档。

**非目标（Out of scope）**：

- 不改动 CSV、PNG、tensorboard、checkpoint、eval_logs 等既有工件的路径与格式。
- 不接管 SB3 内部 logger 的指标记录（tensorboard/CSV/JSON 仍由 SB3 自行写入）。
- 不做将 stderr 一并镜像的功能。

## 2. 用户已确认的设计决策

| 决策点   | 结论                                                         |
| -------- | ------------------------------------------------------------ |
| 实现方向 | 标准 `logging` 重构（非 CSV 美化、非逐回合输出、非全量 tee） |
| 输出目标 | 终端 + 自动归档文件                                          |
| 覆盖范围 | 全部项目自身代码；SB3 回调文本尽量接管                       |
| 文件组织 | `train` 每次启动按时间戳新建独立文件（多次训练并存、不覆盖） |
| 实现方案 | 方案 A：logging 统一 + stdout 镜像补捕获 SB3                 |

## 3. 设计

### 3.1 新增模块 `src/mad_mario/logging/setup.py`

提供两个公共功能：

#### `setup_logging(level: int = logging.INFO, log_dir: Path | None = None, prefix: str = "train") -> Path | None`

装配 root logger。**可重入**：先清空 root 既有 handlers 再装配（避免测试 / 重复调用污染）。

- 固定添加 `StreamHandler` → `sys.stderr`（终端可见）。
- 当 `log_dir` 给定：自动 `mkdir(parents=True, exist_ok=True)`，追加 `FileHandler` → `log_dir / f"{prefix}_{YYYYMMDD_HHMMSS}.log"`（文件名精确匹配 `prefix_\d{8}_\d{6}.log`）。
- 返回日志文件 `Path`；无文件输出时返回 `None`。
- 文件打开失败（磁盘满/权限）→ 降级为仅终端，并记一条 warning，**不得**中断训练。

日志格式（统一 Formatter）：

```
%(asctime)s [%(levelname)s] %(name)s: %(message)s
```

例：`2026-09-06 15:04:22 [INFO] mad_mario.trainer: 创建 8 个并行环境...`

#### `StdoutTee` 与 `capture_stdout_to(log_path: Path)` 上下文管理器

- `StdoutTee(real_stdout, file_path)`：一个 `TextIO`，`write(s)` 时同时写真实 stdout 与追加写日志文件；每次 `write` 后主动 `flush`（避免非终端时 stdout 块缓冲导致 SB3 文本迟到 / 顺序错乱）。
- `capture_stdout_to`：`try/finally` 中把 `sys.stdout` 替换为 Tee，退出时还原。日志文件句柄按路径以追加模式各自持有。

### 3.2 终端 + 文件防重复机制

- 项目自身 logging 消息：`StreamHandler` → **stderr**（不进 stdout），另经 `FileHandler` 直接写日志文件。
- SB3 `EvalCallback` / `CheckpointCallback`（`verbose=1`）的 `print` → **stdout**，被 `StdoutTee` 镜像进同一日志文件。
- 两条来源（stderr 的 logging / stdout 的 SB3）互不重叠 → 文件是两者拼接的完整日志，**无重复**。

### 3.3 各文件改动

#### `src/mad_mario/cli.py`

- `main()`：
  - `train` 分支：先 `config = config_from_train_args(args)`，再 `setup_logging(level=..., log_dir=config.artifacts.save_root / "logs")`，进入 `train()`。
  - `play` 分支：`setup_logging()`（仅终端，不写文件）。
  - `level` 来自 `args.log_level`（见 3.5）。
- `play()` 全部 `print` → 模块级 `logger.info / logger.warning / logger.error`：
  - checkpoint 不存在 → `logger.error`。
  - 关卡标题 / 步数奖励通关结果 / 模型加载提示 → `logger.info`。

#### `src/mad_mario/training/trainer.py`

- 签名扩展：`def train(config: AppConfig, log_path: Path | None = None)`。`log_path` 为 stdout 镜像目标；`None` 时跳过镜像（`capture_stdout_to(None)` 为空操作），便于不以 CLI 方式直接调用 `train` 时保持可用。
- 全部 `tqdm.write` → `logger.info`（“创建 N 个并行环境”“正在恢复 checkpoint”“从头开始创建”“开始训练”“检测到键盘中断…”“最终模型已保存”“训练完成”）；checkpoint 加载失败 → `logger.warning`。
- 移除 `tqdm` import。
- stdout 镜像范围：`env` 与 `eval_env` 创建**之后**（避免子进程在 Tee 内 fork），用 `capture_stdout_to(log_path)` 包住 `model.learn(...)`（含 `KeyboardInterrupt` 处理）至最终保存结束。learn 之前的项目消息由 `FileHandler` 覆盖。

#### log_path 注入链路（明确）

1. `cli.main()` `train` 分支：`config = config_from_train_args(args)`；`log_path = setup_logging(level=..., log_dir=config.artifacts.save_root / "logs")`；调用 `train(config, log_path=log_path)`。
2. `cli.main()` `play` 分支：`setup_logging()`（仅终端，无文件），`play(config_from_play_args(args))` 内部消息走 `logger`。
3. `play()` 函数保持单参签名不变，不需要 `log_path`。

> 以库方式（非 CLI）直接调用 `train(config)` 且未先 `setup_logging` 时，遵循 Python logging 惯例：输出由调用方配置；未配置则仅默认错误级别可见。CLI 是本功能的主路径。

#### `src/mad_mario/logging/metrics.py`

- `MarioMetricsCallback._record()` 的 `tqdm.write` 指标行 → `logger.info`（消息文案与字段不变）。
- 移除 `tqdm` import。
- **CSV / PNG 逻辑完全不动**。

### 3.4 错误与边界

- `setup_logging` 创建目录失败 / 文件打开失败 → 降级仅终端 + warning，不抛异常中断训练。
- `capture_stdout_to` 退出时无论正常或异常均还原 `sys.stdout`。
- `KeyboardInterrupt` 保存流程保留；最终保存后 Tee 还原、logger 正常 flush（Python 退出自动 flush）。
- 配置类 `SystemExit`（config 文件不存在、JSON 解析失败等）行为不变，不经过新日志体系。

### 3.5 `--log-level` 与配置

- `add_train_args` 新增 `--log-level`，`choices=("debug","info","warning","error")`，默认 `"info"`。
- `config_from_train_args` 将其映射为 `SB3ArtifactConfig.log_level`（`str`，新增字段，默认 `"info"`）。
- 仅加在 `train` 子命令（文件日志属于训练场景）；`play` 使用默认 INFO。

### 3.6 日志文件布局

```
<save_root>/              # 如 checkpoints/1-1-v0_right_only（compatible_save_root 结果）
├── logs/
│   └── train_20260906_150422.log
├── latest_metrics.csv
├── latest_reward.png
├── tensorboard/
└── ...
```

### 3.7 测试（新增 `tests/test_logging.py`）

| 用例                                       | 断言                                                                                |
| ------------------------------------------ | ----------------------------------------------------------------------------------- |
| `setup_logging(log_dir=tmp)` 后写一条 info | 日志文件被创建、文件名匹配 `train_\d{8}_\d{6}.log`、内容含该消息                    |
| 不传 `log_dir`                             | 返回 `None`、不创建任何文件                                                         |
| `log_level` 过滤                           | `WARNING` 级别下 info 消息不落盘                                                    |
| `capture_stdout_to(log_path)`              | 期间 `print(...)` 文本同时出现在文件与真实 stdout；退出后 `sys.stdout` 还原为原对象 |

> 依赖 `setup_logging` 的可重入性隔离全局 logger 污染；测试用例 `teardown` 重置 logger。

### 3.8 不破坏的保证

- `latest_metrics.csv` / `latest_*.png` / `tensorboard/` / `snapshots/` / `best_model/` / `eval_logs/` 路径与格式不变。
- SB3 各回调 `verbose` 保持现状（DQN `verbose=0`；Eval/Checkpoint `verbose=1`，文本由 stdout 镜像归档）。
- 现有 `tests/test_config.py`、`tests/test_cuda.py` 不受影响。

## 4. 风险与对策

| 风险                                             | 对策                                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------ |
| stdout 镜像与 logging 文件重复写入               | 项目消息走 stderr + FileHandler；SB3 走 stdout + Tee，来源隔离（见 3.2） |
| 子进程（SubprocVecEnv）在 Tee 内 fork 产生脏输出 | 镜像范围在 env 创建之后才开启                                            |
| 全局 logger 被测试污染                           | `setup_logging` 可重入，先清空再装配                                     |
| 磁盘写失败拖垮训练                               | 文件打开失败降级仅终端，不抛异常                                         |
