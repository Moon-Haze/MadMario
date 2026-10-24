# MadMario · Agent 指南

Super Mario Bros 的 DQN 训练/播放实现（Stable-Baselines3 + gym-super-mario-bros）。`src/` 布局，导入一律 `from mad_mario...`。

**行为真相源**：命令行选项、输出目录结构、训练与播放的环境语义差异，全部写在 `README.md`。本文件不复述它们，只说"东西在哪、哪些契约不能破"。

## 命令

```bash
uv sync --extra cpu                          # 依赖（CUDA 版本见 README 的 extra 表格）
uv run pytest tests -q                       # 全量测试
uv run pytest tests/test_config.py -q        # 单文件
uv run python -m mad_mario.cli train --timesteps 1000   # 冒烟训练
uv run python -m mad_mario.cli play --checkpoint <model.zip> --no-show
```

测试不跑真实训练、不做长录制：假件从 `tests/conftest.py` 的 `MockMarioEnv` 与 `tmp_path` 取。

## 代码地图

| 模块 | 职责 |
| --- | --- |
| `cli.py` | 入口：先扫 `--config` 再解析，再分派 `train`/`play` |
| `config.py` | 配置的**形状**：dataclass + argparse + 装配器，不碰磁盘 I/O |
| `artifacts.py` | 磁盘发现与目录命名：兼容目录名、checkpoint 解析 |
| `env/factory.py` | 环境栈组装：wrapper 顺序只在这里 |
| `env/wrappers.py` | 奖励塑形与卡住处理的 wrapper |
| `training/trainer.py` | DQN 装配、回调、训练循环 |
| `training/best_playback.py` | 达标最优模型的后台录制协议 |
| `playback/runner.py` | 播放会话 `PlaybackSession`：计划、串行/并发派发、命名落盘 |
| `playback/video.py` / `playback/viewer.py` | 录 mp4 / 实时窗口 |
| `logging/metrics.py` / `logging/setup.py` | CSV 指标与图表 / 日志与 stdout 镜像 |

新代码放进上表已有的模块；引入跨模块的新知识前先定归属：配置形状进 `config.py`，磁盘策略进 `artifacts.py`，环境栈进 `env/factory.py`。

## 不能破的契约

- **播放栈单点定义**：`config.playback_env_config()` 决定"播放语义"——关掉奖励裁剪、进度/得分/通关奖励与卡住惩罚，保留卡住脱困，观测变换与训练一致。CLI 播放与训练内最优录制都只调它；任何"播放时关掉 X"的需求加在这里，不要在新调用点改布尔。
- **依赖方向**：`artifacts.py` 只在 `TYPE_CHECKING` 下引用 `config` 的类型。反向 import 会在 import 时成环。
- **续训超参**：调用方显式值 > checkpoint 数据 > 类默认（`trainer._resume_hyperparams`）。结构参数（`policy`/`policy_kwargs`）故意不覆盖——改了网络权重就作废，要换结构就重开一次训练。
- **ε 退火**：`exploration_fraction` 是相对总步数的比例，会随 `--timesteps` 一起放大；`exploration_steps` 是绝对预算，续训时按全局步数重算。需要固定退火窗口就用绝对预算。
- **采样播放**：探索率来自 checkpoint，在 `[0, 2ε]` 上均匀抽；目录名承载探索率（点改横杠），文件名只留关卡。加载后要把 RNG 种子交还 OS 熵，否则每遍逐帧相同。
- **产物不覆盖**：每次播放开新的时间戳目录，同秒追加序号；同目录内撞名顺延 `_2`。
- **最优录制**：后台线程 + 队列，训练收尾前必须 drain——mp4 的索引在文件尾部，没收尾的片子谁都打不开。录制失败只告警，不中断训练，也不丢排队中的录制。
- **后端契约**：训练路径的 matplotlib Agg 由 `logging/metrics.py` 自己保证，`cli.py` 不承担导入顺序责任。
- **并发播放**：无窗口时线程数 = CPU 核数/2，每线程自备模型副本——`exploration_rate` 会被各趟改写，共享实例会让目录名失真。

## 变更流程

项目走 OpenSpec 规格驱动（`.agents/skills/` 里的 `/opsx:*` 命令）：

1. 改行为 → `/opsx:propose`，产出 proposal / design / specs / tasks（落在 `openspec/changes/<name>/`）
2. 实现 → `/opsx:apply`，按 tasks 逐条落地
3. 收尾 → `/opsx:archive`，把 delta spec 并入 `openspec/specs/` 并归档 change

`openspec/specs/` 是行为规格的真相源（当前有 `playback`、`best-run-recording`）；动播放或最优录制的行为，先改对应 spec 再改代码。`docs/superpowers/` 存历史设计文档，只作背景参考，不作约束。

## 约定

- 注释、docstring、面向用户的日志用中文；注释写**为什么**（含踩过的坑：内存估算、ε 语义、覆盖风险），不重述代码在做什么。
- 公共函数带类型注解；配置用 dataclass，字段默认值即 CLI 默认值——`config.py` 的 argparse `default` 直接引用 `EnvConfig.x` / `SB3Config.x`，改一处即可。
- 同一组语义收敛到单点定义，调用方指名调用，不各自复述。
- 提交信息用中文 Conventional Commits：`feat:` / `fix:` / `refactor:` / `chore:` / `docs:`，破坏性改动标 `refactor!:`。

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
