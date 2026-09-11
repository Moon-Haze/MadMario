# 卡住脱困机制：从"判负重开"升级为"随机接管脱困"

- 日期：2026-09-11
- 状态：待评审（brainstorming 设计已获确认，两项关键决策：主动干预脱困 + 均匀随机扰动）
- 范围：`mad_mario.env.wrappers` / `mad_mario.env.factory` / `mad_mario.config`

## 1. 背景与目标

角色会因障碍物卡在一个位置无法移动。现有 `StuckPenalty` 只能"判负结束回合"，不做任何脱困动作，且播放模式下被显式关闭，导致观看时角色原地站到游戏内计时耗尽。

**目标**：角色卡住时主动接管动作、随机尝试脱困；脱困成功则交还控制权继续本回合，尝试用尽才判负。训练、评估、播放三条路径全部生效。

**本设计由两项已确认的决策约束**：

1. 卡住时的目标行为是**主动干预脱困**（而非仅调参、仅惩罚、或仅在播放模式兜底）。
2. 脱困动作采用**均匀随机扰动**（从动作空间均匀采样，不偏向跳跃、不做分级递增）。

## 2. 非目标（Out of scope）

- **不改动作空间预设**。`right_only` 无 `left` 动作（`RIGHT_ONLY` 实为 `[NOOP, right, right+A, right+B, right+A+B]` 共 5 个），因此"后退脱离"在本期无法实现；改用 `simple`/`complex` 会使旧 checkpoint 的动作层维数不匹配（5→7 或 12）而作废已有训练成果，明确排除。
- 不新增训练指标（救援次数、救援成功率等）。本期只写 `info` 字段，不改 `MarioMetricsCallback`、CSV 列、PNG 图表。
- 不修改 DQN 超参、网络结构、checkpoint 格式、回放缓冲区行为。
- 不拦截或过滤救援期间产生的转移——它们照常进入 SB3 回放池。
- 不改造"卡住"以外的奖励通道（`ProgressReward` / `ScoreReward` / `FlagReward` 不动）。
- 不改写历史设计文档 `2026-06-21-stuck-penalty-training-design.md`。

## 3. 现状与四个缺陷

现有实现：`src/mad_mario/env/wrappers.py:6-65` 的 `StuckPenalty`，构造于 `src/mad_mario/env/factory.py:44-49`，配置位于 `src/mad_mario/config.py:30-32`，CLI 位于 `src/mad_mario/config.py:410-413`。

### 缺陷 1：不做脱困动作

触发后仅 `reward -= penalty`、`truncated = True`。`play` 模式使用 `deterministic=True`（`src/mad_mario/cli.py:109`），回合重开后确定性策略面对同一障碍会选同样的动作，反复卡在同一个点。

### 缺陷 2：抖动漏判（本次修复的核心）

判定是"相邻两步 `x_pos` 与 `y_pos` 均未变化"（`wrappers.py:37-51`）。而"贴着障碍反复被顶回来"的形态是 `x` 在 `100 → 101 → 100 → 101` 之间抖动——每一步都被判为"移动了"，`stuck_steps` 持续清零，阈值**永远不会达成**。因此只加干预而不改判定，在用户描述的场景里干预永远不会触发。

### 缺陷 3：原地跳跃被误判为"未卡住"

`y_pos` 变化被计为"移动"。角色在障碍前原地反复跳跃、`x` 毫无推进时，计数被 y 的变化无限重置。这正是需要干预的典型形态之一。

### 缺陷 4："卡住"不被当作真终止

SB3 的 `SubprocVecEnv` worker 会把包装器设置的 `truncated=True` 转写为 `info["TimeLimit.truncated"] = True`（`.venv/.../stable_baselines3/common/vec_env/subproc_vec_env.py:38`），缓冲区据此记录 `timeouts`（`common/buffers.py:278`），DQN **仍然从下一状态 bootstrap** Q 值。因此 `-5.0` 是唯一有效信号，值函数不把"卡住"视为死路。本设计不改变这一机制（改它会动到 bootstrap 语义，超出范围），但需在文档中明确。

## 4. 检测口径：窗口内净推进

替换原有的"相邻两步对比"，改为**基于 `x` 的滑动窗口净推进判定**，只看水平推进，不看 `y`。

### 状态量

| 字段 | 含义 |
| --- | --- |
| `best_x` | 历史最大 `x`，只增不减；对外即 `info["max_x_pos"]` |
| `window_start_x` | 当前停滞窗口的起点基准 |
| `window_best_x` | 窗口内见过的最大 `x`，只增不减 |
| `stuck_steps` | 自窗口开始以来的步数 |
| `rescue_steps` | 救援已消耗的步数 |
| `rescuing` | 是否处于救援阶段 |

### 判定规则

每步读取 `x = info.get("x_pos")`：

- `x` 可用时：`best_x = max(best_x, x)`、`window_best_x = max(window_best_x, x)`。
- **取得进展** 当且仅当 `x` 可用且 `window_best_x - window_start_x >= progress_epsilon`。
  - 取得进展 → `window_start_x = window_best_x`、`stuck_steps = 0`、退出救援（`rescuing = False`、`rescue_steps = 0`）。
  - 未取得进展 → `rescuing` 为真则 `rescue_steps += 1`，否则 `stuck_steps += 1`。
- `x` 不可用时**不做任何判定**，只保留已知诊断量（沿用旧设计"缺失位置信息不得触发误判"的约束）。

### 为什么这样能堵住抖动漏判

窗口基准 `window_start_x` 只在跨越阈值时才前移，因此窗口内 `window_best_x` 的上界被钉在 `window_start_x + progress_epsilon` 以内。`100 ↔ 101` 抖动时差值恒为 1 < 8 → `stuck_steps` 持续累加 → 正确触发。正常向右前进时每步推进几十像素，差值很快达标 → 计数清零。

### 默认阈值

```python
progress_epsilon = 8.0   # 像素，约半格
```

单位是 Mario 的 `x_pos` 像素。8.0 表示"窗口内必须有半格以上的净前进才算取得进展"。

### 已知取舍

只看 `x` 意味着策略若学会了"原地等待敌人/等待移动平台"，等待超过 `max_stuck_steps`（默认 120 步 × `frame_skip=4` = 480 帧 ≈ 8 秒游戏时间）也会被干预打断。判定为可接受代价：随机动作集中的 `NOOP` 本身就能实现等待，且救援结束会交还控制权。

## 5. 状态机与救援行为

```text
NORMAL  --(stuck_steps >= max_stuck_steps，且 rescue_enabled)------> RESCUE
NORMAL  --(stuck_steps >= max_stuck_steps，且 !rescue_enabled)-----> 判负结束（旧行为）
RESCUE  --(取得进展，净推进 >= progress_epsilon)-------------------> NORMAL   (stuck_steps 清零)
RESCUE  --(rescue_steps >= rescue_max_steps，且 penalty_enabled)---> 判负结束
RESCUE  --(rescue_steps >= rescue_max_steps，且 !penalty_enabled)---> RESCUE   (开新一轮，不截断)
```

- **进入 RESCUE**：当步判定之后置位，因此**下一步**的动作才被接管。进入时 `rescue_steps = 0`，`stuck_steps` 保持不变（保留"触发救援时的停滞步数"这一诊断量）。
- **RESCUE 期间**：
  - `step(action)` 在调用内层环境**之前**丢弃传入的 `action`，替换为 `self.action_space.sample()`（对 `right_only` 即 5 个动作均匀随机）。这是 gym wrapper 的标准用法（`StickyAction`、`DelayObservation` 等经典 wrapper 同样在 `step()` 内改写动作），也是 `play` 模式下确定性策略能被推动的唯一途径。
  - 不扣分、不截断，改用 `rescue_steps` 计数，给救援一个干净的尝试窗口。
  - 随机动作产生的转移照常进入回放池（明确接受的副作用）。
- **退出 RESCUE**：取得进展即退出并交还控制权。"取得进展"复用同一套窗口判定，不引入第二套阈值。
- **救援失败**：`rescue_steps >= rescue_max_steps` 仍无进展 → 按 `penalty_enabled` 分流（见下表）。

### 两个开关的完整行为矩阵

| `rescue_enabled` | `penalty_enabled` | 达到 `max_stuck_steps` 时 | 救援预算耗尽时 |
| --- | --- | --- | --- |
| True | True | 进入救援 | 扣分 + 截断 |
| True | False | 进入救援 | **开始新一轮救援**（不扣分、不截断） |
| False | True | 扣分 + 截断（与改动前一致） | — |
| False | False | 不注入包装器，零开销 | — |

两个推导出的后果，均需显式记录：

1. **训练模式下的截断时机被推后**：`rescue_enabled=True, penalty_enabled=True` 时，一个毫无进展的回合会在 `max_stuck_steps + rescue_max_steps`（默认 240 步）后才截断，而非原来的 120 步。这是"先容忍停滞、再给救援窗口"的自然结果，但会改变回合长度分布。
2. **"救援失败后开始新一轮"是显式设计，不是副作用**。若不这样做，`rescuing` 转假后下一步会因 `stuck_steps` 仍等于 `max_stuck_steps` 而立刻重新进入救援，其间**插入一步策略动作**——行为上仍是"永不站住"，但语义含混、且那一步策略动作是浪费。因此救援失败且不截断时**原地**重置 `rescue_steps = 0` 并保持 `rescuing = True`，形成无缝的连续随机挣扎。

### 播放模式的组合语义

`play` 使用矩阵第 2 行（`rescue_enabled=True, penalty_enabled=False`）：**角色卡住就自己连续随机挣扎，且永远不会因为卡住而退出关卡**，直到被敌人击杀或游戏内计时耗尽自然结束。这正是需求描述的期望行为。

`stuck_penalty_enabled` 因此同时控制"扣分"与"提前结束"，语义需在配置字段注释中写明。

需提醒的副作用：训练时若用户只传 `--no-stuck-penalty` 而不传 `--no-stuck-rescue`，等价于矩阵第 2 行——卡住的回合会无限随机挣扎，可能显著拉长该回合并污染回放池。这是用户主动选择的组合。

### 抑制条件

以下情况不进入救援、也不判定卡住：内层环境已 `terminated` 或 `truncated`（沿用旧约束），或 `info["flag_get"]` 为真（拉旗后的滑杆/走向城堡动画期间 `x` 可能长时间不增，属正常通关流程）。

## 6. info 字段与语义变更

| 字段 | 变化 |
| --- | --- |
| `info["max_x_pos"]` | 保留，值来自 `best_x`。**当前 `src/` 中无生产消费方**：`metrics.py:103` 读的是 `info["x_pos"]` 并自行累计 `_max_x`。保留该键是为向后兼容外部工具；真正不可删的是 `info["x_pos"]` |
| `info["stuck_steps"]` | 保留，含义为 NORMAL 阶段下的停滞步数 |
| `info["rescuing"]` | 新增，`bool` |
| `info["rescue_steps"]` | 新增，`int` |
| `info["stuck"]` | **语义变更**：原先"达到停滞阈值即为 True"，现在仅在"救援失败被截断"时为 True |

语义变更的安全性已核实：全仓库检索确认 `info["stuck"]` 只被 `tests/test_wrappers.py` 消费，`metrics.py` 不读取它，无生产代码依赖旧语义。

`info` 在 `step()` 与 `reset()` 中均先 `dict(info)` 复制再写入，避免污染向量环境复用的字典（沿用旧设计约束）。

## 7. 配置与 CLI

### EnvConfig 新增字段

```python
stuck_progress_epsilon: float = 8.0     # 窗口内净推进阈值（像素）
stuck_rescue_enabled: bool = True       # 卡住后随机接管动作
stuck_rescue_max_steps: int = 120       # 救援预算，用尽则按惩罚开关处理
```

已有字段 `stuck_penalty_enabled=True`、`stuck_max_steps=120`、`stuck_penalty=5.0` 保持不变。

### CLI 新增选项

新增到现有奖励参数组（`src/mad_mario/config.py:410-413` 旁）：

```text
--no-stuck-rescue
--stuck-rescue-max-steps
--stuck-progress-epsilon
```

现有 `--no-stuck-penalty` 的语义收窄为"只关惩罚与截断，不关救援"。要与今天完全一致的行为，需同时传 `--no-stuck-penalty --no-stuck-rescue`（此时不注入包装器，零开销）。此语义变化需在 `--no-stuck-penalty` 的 help 文本中写明。

### 播放路径

`config_from_play_args()`（`src/mad_mario/config.py:536-541`）改为：

```python
stuck_penalty_enabled=False,
stuck_rescue_enabled=True,
```

`stuck_max_steps`、`stuck_rescue_max_steps`、`stuck_progress_epsilon` 用 `EnvConfig` 默认值。

## 8. 架构决策：扩展单类，不拆分包装器

### 决策

把 `StuckPenalty` 重命名为 `StuckHandler`（它现在承担检测、惩罚、救援三件事，旧名字会说谎），在类内实现状态机。**不**新增独立的 `ActionWrapper` 子类。

### 理由

gymnasium 官方的 `ActionWrapper` 在职责划分上确实更"单一"（专门用于改写动作）。否决它的决定性理由：**救援与截断必须互斥协调**——救援进行中不能截断，救援失败才截断。若拆成 `StuckRescue` 与 `StuckPenalty` 两个类，就需要 `StuckRescue` 读取对方的停滞计数器、对方读取 `StuckRescue` 的救援状态，形成**双向跨层依赖**，比同类的两个分支复杂得多，且包装器顺序（`StuckRescue` 必须在外层）会成为后续改动的隐性陷阱。

现有 4 个奖励包装器（`ProgressReward`/`ScoreReward`/`FlagReward`/`StuckPenalty`）都是各自独立、无共享状态；本次是首个需要共享状态的场景，同包内聚更合适。

### 连带清理

删除死参数 `movement_epsilon`（`wrappers.py:7,11`）。全仓库检索确认 `factory.py` 与测试均未传该参数，始终取默认值 `0.0`；新口径不再有"位置变化量"这一概念，由 `progress_epsilon` 取代。

### 改名影响面

- `src/mad_mario/env/factory.py:10`（import）与 `:45`（构造）
- `tests/test_wrappers.py:10`（import）与 `TestStuckPenalty` 类名及其内部引用

历史设计文档不改写。

## 9. 包装器装配

位置不变（`ClipReward` 之后、观测预处理之前），但注入条件由单一开关改为析取：

```python
if config.stuck_penalty_enabled or config.stuck_rescue_enabled:
    env = StuckHandler(
        env,
        max_stuck_steps=config.stuck_max_steps,
        penalty=config.stuck_penalty,
        progress_epsilon=config.stuck_progress_epsilon,
        penalty_enabled=config.stuck_penalty_enabled,
        rescue_enabled=config.stuck_rescue_enabled,
        rescue_max_steps=config.stuck_rescue_max_steps,
    )
```

置于 `ClipReward` 之后保证救援失败时的扣分不被裁剪回 `-1`（沿用旧设计约束）。两个开关全为假时不注入包装器，行为与当前实现一致。

### 构造函数签名

```python
def __init__(
    self,
    env,
    max_stuck_steps,
    penalty,
    progress_epsilon=8.0,
    penalty_enabled=True,
    rescue_enabled=True,
    rescue_max_steps=120,
):
```

## 10. 边界与错误处理

- `x_pos` 缺失（`None`）时不做判定、不累加任何计数、不进入救援；`info["max_x_pos"]` 保持上一次已知值。
- `reset()` 时若 `x_pos` 缺失，`best_x`、`window_start_x`、`window_best_x` 均初始化为 `0.0`（沿用旧实现"缺失位置则记 0"的处理）。后续首次拿到真实 `x` 时，`window_best_x - window_start_x` 可能一次性超过阈值而判定为"取得进展"——这只会清零计数，无副作用，明确接受。
- 内层已 `terminated` 或 `truncated` 时不覆盖其状态，只在救援失败分支且 `penalty_enabled` 为真时设置截断。
- `info["flag_get"]` 为真时跳过判负与救援进入（停滞计数仍照常维护，只是不触发任何处置）。
- `rescue_max_steps` 下限钳制为 1（与 `stuck_max_steps` 一致，经由 `max(1, ...)` 在 `config_from_train_args` 中处理）。
- **已知限制**：救援使用环境动作空间的 RNG，不受 SB3 `seed` 控制（`SubprocVecEnv` 创建时未显式播种，`AsyncVectorEnv`/`SubprocVecEnv` 由 SB3 以无 seed 方式重置）。因此救援动作序列不可复现。本期不解决——修复它需要改动向量环境的播种链路，超出范围。

## 11. 测试与验证

沿用 `tests/test_wrappers.py` 现有的 `_BaseMockEnv` 模式（可控制 `_make_info()` 输出的最小 mock 环境）。

### 新增测试

1. **抖动不重置**：`x` 在 `100 ↔ 101` 间交替、`progress_epsilon=8.0`、`max_stuck_steps=3` → `stuck_steps` 必须累加到 3 并进入救援。**现有实现下此测试必然失败**，是本次修复的核心证据。
2. **仅 y 变化仍计入卡住**：`x` 恒定、`y` 每步变化 → 仍进入救援。
3. **救援接管动作**（须确定性，不用概率断言）：用 `monkeypatch` 把 `env.action_space.sample` 替换为固定返回 `3` 的函数，调用方传入 `0`；mock 内层环境记录收到的 action，断言进入救援后收到的是 `3` 而非 `0`。
4. **救援中取得进展即退出**：救援期间 `x` 推进 ≥ 8 → `rescuing` 转假、`stuck_steps` 归零、不截断、奖励不变。
5. **救援预算耗尽且惩罚开启**：连续无进展满 `rescue_max_steps` → `reward == -penalty`、`truncated is True`、`info["stuck"] is True`。
6. **救援预算耗尽且惩罚关闭**：`penalty_enabled=False` → 不截断、不扣分、`info["stuck"]` 不置位，且 `rescuing` 保持为真、`rescue_steps` 归零（无缝进入新一轮救援，而非插入一步策略动作）。
7. **`--no-stuck-rescue` 保持旧截断行为**：`rescue_enabled=False, penalty_enabled=True` → 达到 `max_stuck_steps` 即扣分并截断，不进入救援。
8. **`flag_get` 抑制**：`info["flag_get"]=True` 且 `x` 恒定 → 不进入救援。
9. **`x_pos` 缺失不误判**：`info` 无 `x_pos` → 永不进入救援。
10. **`max_x_pos` 语义保持**：仍等于历史最大 `x`，回退时不变小。

### 需改写的既有测试

`tests/test_wrappers.py` 的 `TestStuckPenalty` 随类改名同步为 `TestStuckHandler`。其中 `test_stuck_triggers` 断言"第 3 步截断且 `reward == -5.0`"。新语义下第 3 步只是**进入救援**，不再截断。改写为断言进入救援（`info["rescuing"] is True`、奖励不变、未截断），截断断言移交新增测试 5。

`test_not_stuck_moves`、`test_stuck_recovers`、`test_stuck_tracks_max_x` 需适配新构造签名（`test_stuck_recovers` 的"移动后计数重置"改为"推进 ≥ `progress_epsilon` 后计数重置"）。

### 配置测试

`tests/test_config.py` 新增：CLI 接受 `--no-stuck-rescue`、`--stuck-rescue-max-steps`、`--stuck-progress-epsilon` 并正确映射到 `EnvConfig`；`config_from_play_args()` 下 `stuck_rescue_enabled is True` 且 `stuck_penalty_enabled is False`。

### 验证命令

```bash
uv run pytest tests/test_wrappers.py tests/test_config.py -v
uv run pytest -q
```

## 12. 成功标准

1. `x` 在 ±1 像素抖动满阈值后进入救援（现有实现在此场景不会触发）。
2. 救援期间下发到内层环境的动作来自 `action_space.sample()`，与策略输出无关。
3. 救援中净推进 ≥ `progress_epsilon` 即退出救援，本回合继续，奖励不受影响。
4. 救援满 `rescue_max_steps` 无进展时：`penalty_enabled=True` 扣 `stuck_penalty` 并截断；`penalty_enabled=False` 不扣分、不截断、无缝开始新一轮救援。
5. `rescue_enabled=False, penalty_enabled=True` 时达到 `max_stuck_steps` 即扣分截断，与改动前行为一致。
6. `--no-stuck-penalty --no-stuck-rescue` 同时传入时不注入包装器，行为与改动前一致。
7. `uv run pytest -q` 全量通过。
