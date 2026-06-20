# MadMario 包结构重构设计

## 背景

当前项目已经拆出了 `agent.py`、`trainer.py`、`config.py`、`env_factory.py`、`metrics.py`、`replay_buffer.py` 等模块，但代码仍平铺在仓库根目录。训练入口分散在 `main.py` 和 `main_vector.py`，checkpoint 保存、metrics 输出、训练循环、智能体状态管理之间仍有耦合。

本次重构目标是提升代码的可扩展性和可阅读性，同时支持更干净的 checkpoint/曲线输出策略。

## 目标

1. 将根目录脚本重组为标准 Python 包 `mad_mario/`。
2. 使用统一 CLI 替代 `main.py` 和 `main_vector.py`。
3. 将训练配置、环境配置、智能体配置、输出配置集中管理。
4. 将 checkpoint 保存逻辑从 `Mario` 智能体中拆出。
5. 支持默认只保留 latest 输出，同时可选保留 runs 历史目录。
6. 保持训练循环清晰，避免过度抽象。
7. README 同步更新为新入口和新目录说明。

## 非目标

1. 不在本次重构中更换算法类型。
2. 不修改神经网络结构。
3. 不修改环境 wrapper 顺序。
4. 不引入复杂实验管理框架或外部配置系统。
5. 不保留旧入口作为主要使用方式。

## 推荐架构

```text
mad_mario/
  __init__.py
  cli.py
  config.py

  agent/
    __init__.py
    mario.py
    replay_buffer.py
    checkpoint.py

  env/
    __init__.py
    factory.py
    wrappers.py

  models/
    __init__.py
    mario_net.py

  training/
    __init__.py
    trainer.py
    loops.py
    progress.py
    artifacts.py

  logging/
    __init__.py
    metrics.py
```

## CLI 设计

统一使用包入口：

```bash
python -m mad_mario.cli train
python -m mad_mario.cli train --vector
python -m mad_mario.cli play
```

可选在 `pyproject.toml` 中增加 console script：

```bash
mad-mario train
mad-mario train --vector
mad-mario play
```

训练参数包括：

- `--episodes`
- `--vector`
- `--num-envs`
- `--checkpoint`
- `--no-resume`
- `--keep-runs` / `--no-keep-runs`
- `--save-root`
- `--record-every`
- `--save-every`
- `--batch-size`
- `--burnin`
- `--learning-rate`
- `--gamma`
- `--exploration-decay`
- `--exploration-min`

默认行为：

- `train` 默认使用单环境。
- 加 `--vector` 后使用并行环境。
- 默认自动从 `checkpoints/latest.chkpt` 恢复。
- 默认 `keep_runs=False`，只维护 latest 输出。
- 需要历史目录时显式传入 `--keep-runs`。

## 配置设计

`mad_mario.config` 中使用 dataclass 管理配置：

- `TrainingConfig`
  - `episodes`
  - `batch_size`
  - `burnin`
  - `learn_every`
  - `sync_every`
  - `record_every`
  - `num_envs`
  - `vector`

- `AgentConfig`
  - `gamma`
  - `learning_rate`
  - `exploration_rate`
  - `exploration_rate_decay`
  - `exploration_rate_min`
  - `replay_capacity`

- `ArtifactConfig`
  - `save_root`
  - `resume`
  - `checkpoint`
  - `keep_runs`
  - `run_name`
  - `save_every`

- `EnvConfig`
  - `env_name`
  - `state_dim`
  - `frame_skip`
  - `frame_stack`
  - `resize_shape`
  - `render_mode`

## 模块职责

### `mad_mario.agent.mario`

`Mario` 只负责智能体行为和算法状态：

- `act()` / `act_batch()` 选择动作。
- `cache()` / `cache_batch()` 写入经验。
- `learn()` 执行学习调度。
- `sync_Q_target()` 同步目标网络。
- `state_dict()` 导出训练状态。
- `load_state_dict()` 恢复训练状态。

`Mario` 不再直接处理文件路径，也不负责写 checkpoint 文件。

### `mad_mario.agent.checkpoint`

新增 `CheckpointManager`，负责：

- 从指定 checkpoint 或 latest checkpoint 加载。
- 保存 latest checkpoint。
- 在 `keep_runs=True` 时额外保存当前 run 内 checkpoint。
- 统一 checkpoint 字段格式。
- 打印加载和保存信息。

### `mad_mario.training.artifacts`

新增 `RunArtifacts`，负责统一管理输出路径。

默认输出：

```text
checkpoints/
  latest.chkpt
  latest_metrics.csv
  latest_reward_plot.png
  latest_length_plot.png
  latest_loss_plot.png
  latest_q_plot.png
```

启用 `--keep-runs` 后额外输出：

```text
checkpoints/runs/<timestamp>/
  checkpoint.chkpt
  metrics.csv
  reward_plot.png
  length_plot.png
  loss_plot.png
  q_plot.png
```

run 目录内也只保留一个 `checkpoint.chkpt`，不再生成 `mario_net_step_*.chkpt`。

### `mad_mario.logging.metrics`

`MetricLogger` 继续负责：

- 收集 episode reward、length、loss、q。
- 写 CSV。
- 绘制四张曲线图。

变化是它接收明确的输出路径，而不是只接收一个目录。

### `mad_mario.training.trainer`

负责装配训练所需组件：

- 创建 artifacts。
- 创建 env。
- 创建 agent。
- 创建 checkpoint manager。
- 创建 logger。
- 根据配置选择单环境或并行环境训练循环。

### `mad_mario.training.loops`

保留训练循环主体：

- `train_single_env_loop(...)`
- `train_vector_env_loop(...)`

训练循环不直接关心 checkpoint 路径细节，只在需要保存时调用 `CheckpointManager`。

### `mad_mario.training.progress`

封装 tqdm 进度条和状态字符串，减少训练循环中的显示细节。

## 保存策略

默认只保留最新输出，避免 checkpoint 目录堆积大文件。

启用 `--keep-runs` 后保留当前 run 的 metrics、曲线和 checkpoint，方便之后对比实验。

latest 输出始终更新，runs 输出仅在 `keep_runs=True` 时更新。

## 旧入口处理

用户选择“只用新入口”。因此本次重构将删除或废弃 `main.py` 和 `main_vector.py` 的旧入口角色。README 中将使用新命令作为唯一推荐方式。

如果保留文件，也只保留极短迁移提示，不再承载训练逻辑。

## 行为调整边界

本次允许整理默认参数，但为降低重构风险，算法核心保持不变：

- 不改 Double DQN 目标计算逻辑。
- 不改 CNN 网络结构。
- 不改环境 wrapper 顺序。
- 不改 replay buffer 存储语义。

会做的调整：

- 将 `gamma`、`learning_rate`、`exploration_rate`、`exploration_rate_decay`、`exploration_rate_min` 等常量迁移到配置。
- 将 checkpoint 保存从智能体中移出。
- 将 latest/runs 输出策略集中到 artifact/checkpoint 模块。

## 验证计划

重构后运行：

```bash
python -m py_compile mad_mario/cli.py mad_mario/config.py mad_mario/agent/*.py mad_mario/env/*.py mad_mario/models/*.py mad_mario/training/*.py mad_mario/logging/*.py
```

并尽量执行短训练验证：

```bash
python -m mad_mario.cli train --episodes 1 --no-resume --record-every 1
python -m mad_mario.cli train --vector --num-envs 2 --episodes 2 --no-resume --record-every 1
python -m mad_mario.cli train --episodes 1 --no-resume --record-every 1 --keep-runs
```

检查项：

- `checkpoints/latest.chkpt` 能生成或更新。
- `checkpoints/latest_metrics.csv` 能生成或更新。
- 四张 latest 曲线图能生成或更新。
- `--keep-runs` 时额外生成 `checkpoints/runs/<timestamp>/...`。
- 不再生成 `mario_net_step_*.chkpt`。

## 风险与缓解

1. **import 路径迁移风险**
   - 通过语法检查和短训练验证发现问题。

2. **checkpoint 字段变更风险**
   - 保持现有 checkpoint 的关键字段兼容：`model`、`optimizer`、`exploration_rate`、`curr_step`、`episode`。

3. **训练行为变化风险**
   - 算法核心不改，只移动配置和文件职责。

4. **旧脚本使用习惯变化**
   - README 明确新命令；旧入口不再作为主要入口。

## 完成标准

1. 项目可通过新 CLI 启动单环境和并行训练。
2. checkpoint/metrics/plot 输出符合 latest/runs 策略。
3. 代码职责更清晰，训练循环中不再混杂路径管理细节。
4. `Mario` 类不再直接写文件。
5. README 与实际入口保持一致。
6. 语法检查通过，短训练验证结果明确。