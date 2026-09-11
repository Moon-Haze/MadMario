# 卡住脱困机制 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Mario 因障碍物卡住时主动接管动作、随机尝试脱困，脱困成功即交还控制权，尝试用尽才判负；同时修掉现有检测口径漏判"贴墙抖动"的缺陷。

**Architecture:** 把 `src/mad_mario/env/wrappers.py` 的 `StuckPenalty` 重命名为 `StuckHandler`，把"相邻两步位置对比"替换为"窗口内 x 净推进判定"，并在类内加入 `NORMAL`/`RESCUE` 状态机：进入 `RESCUE` 后忽略传入动作、改用 `action_space.sample()`，取得净推进即退出。两个开关（`stuck_penalty_enabled`、`stuck_rescue_enabled`）决定四种组合的行为，其中播放模式用 `rescue=True, penalty=False` 实现"永不因卡住而结束关卡"。

**Tech Stack:** Python ≥3.13、gymnasium（`gym.Wrapper`、`Discrete`）、gym_super_mario_bros、pytest（uv 工程）。

**Spec:** [docs/superpowers/specs/2026-09-11-stuck-rescue-design.md](../specs/2026-09-11-stuck-rescue-design.md)

## Global Constraints

以下约束对每个任务都生效，逐条照抄自 spec：

- 所有用户可见文本使用简体中文：代码注释、argparse `help` 文本、测试断言消息、提消息均用中文。
- 精确默认值：`progress_epsilon = 8.0`（像素）、`stuck_max_steps = 120`、`rescue_max_steps = 120`、`stuck_penalty = 5.0`。
- 检测**只看 `x_pos`，不看 `y_pos`**（原地反复跳跃而 x 不前进同样属于卡住）。
- 窗口判定：`window_best_x - window_start_x >= progress_epsilon` 即"取得进展"→ `window_start_x = window_best_x`、`stuck_steps = 0`、退出救援。`window_best_x` 只增不减，窗口基准只在跨阈值时才前移。
- `x_pos` 缺失（`None`）时**不做任何判定、不累加任何计数、不进入救援**。
- 救援动作 = `self.action_space.sample()`，必须在 `step()` 内、**调用内层环境之前**替换传入的 `action`。
- 两开关行为矩阵，四条组合必须全部成立：

  | `rescue_enabled` | `penalty_enabled` | 达到 `max_stuck_steps` | 救援预算耗尽 |
  | --- | --- | --- | --- |
  | True | True | 进入救援 | 扣分 + 截断 |
  | True | False | 进入救援 | 无缝开新一轮救援（不扣分、不截断、不插策略动作） |
  | False | True | 扣分 + 截断（与改动前一致） | — |
  | False | False | 不注入包装器 | — |

- `info["stuck"]` **仅**在"救援失败且 `penalty_enabled=True`"（或矩阵第 3 行的阈值直达）时为 `True`；进入救援不置位。
- `info` 必须先在 `reset()` / `step()` 内 `dict(info)` 复制再写入，避免污染向量环境复用的字典。
- 保留 `info["max_x_pos"]` 与 `info["stuck_steps"]`。注意：`info["max_x_pos"]` 当前在 `src/` 中无生产消费方（`metrics.py:103` 读的是 `info["x_pos"]` 并自行累计 `_max_x`），保留是为向后兼容；真正不可删的是 `info["x_pos"]`。
- 内层环境已 `terminated`/`truncated` 时不覆盖其状态；`info["flag_get"]` 为真时跳过卡住判定、救援进入与判负。
- 不改变 `ClipReward`/`ProgressReward`/`ScoreReward`/`FlagReward` 的行为，不改变包装器链中 `StuckHandler` 的位置（仍在 `FlagReward` 之后、`GrayScaleObservation` 之前）。
- 测试命令一律用 `uv run pytest <path> -v`（项目为 uv 工程）。
- 只 `git add` 任务清单列出的文件路径，不要 `git add -A`。

---

### Task 1: 配置层（EnvConfig 字段 + CLI 选项 + train/play 映射）

**Files:**
- Modify: `src/mad_mario/config.py`（`EnvConfig` 约 30-32 行；`add_train_args` 奖励组约 410-414 行；`config_from_train_args` 约 478-480 行；`config_from_play_args` 约 540 行）
- Test: `tests/test_config.py`（追加到文件末尾）

**Interfaces:**
- Consumes: 无（本任务是起点）
- Produces: `EnvConfig.stuck_progress_epsilon: float = 8.0`、`EnvConfig.stuck_rescue_enabled: bool = True`、`EnvConfig.stuck_rescue_max_steps: int = 120`；train 子命令 CLI 选项 `--no-stuck-rescue` / `--stuck-rescue-max-steps` / `--stuck-progress-epsilon`；`config_from_play_args()` 产出 `stuck_penalty_enabled=False` 且 `stuck_rescue_enabled=True`

本任务**不改变任何运行期行为**——`env/factory.py` 此时还没读这些新字段。这是刻意的：先让配置层就位并独立通过测试，再改包装器。

**注意**：`add_play_args()`（`config.py:426-443`）不含奖励参数组，play 侧不新增 CLI 选项，脱困行为由 `config_from_play_args()` 硬编码。

- [ ] **Step 1: 写失败测试**

在 `tests/test_config.py` 末尾追加（文件已 `from __future__ import annotations`，已 import `argparse`、`EnvConfig`、`add_train_args`、`add_play_args`、`config_from_train_args`、`config_from_play_args`）：

```python
# ─────────────────────────────
# 卡住脱困配置
# ─────────────────────────────
class TestStuckRescueConfig:
    def test_env_config_defaults(self):
        cfg = EnvConfig()
        assert cfg.stuck_progress_epsilon == 8.0
        assert cfg.stuck_rescue_enabled is True
        assert cfg.stuck_rescue_max_steps == 120

    def test_cli_flags_map_to_env_config(self):
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(
            [
                "--no-stuck-rescue",
                "--stuck-rescue-max-steps",
                "7",
                "--stuck-progress-epsilon",
                "3.5",
            ]
        )
        config = config_from_train_args(args)
        assert config.env.stuck_rescue_enabled is False
        assert config.env.stuck_rescue_max_steps == 7
        assert config.env.stuck_progress_epsilon == 3.5

    def test_rescue_max_steps_clamped_to_one(self):
        # 与 stuck_max_steps 一致：下限钳制为 1，避免预算为 0 时每步都判负
        parser = argparse.ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args(["--stuck-rescue-max-steps", "0"])
        config = config_from_train_args(args)
        assert config.env.stuck_rescue_max_steps == 1

    def test_play_enables_rescue_without_penalty(self):
        # 播放语义：不扣分、不因卡住提前结束关卡，但必须随机挣扎脱困
        parser = argparse.ArgumentParser()
        add_play_args(parser)
        args = parser.parse_args([])
        config = config_from_play_args(args)
        assert config.env.stuck_penalty_enabled is False
        assert config.env.stuck_rescue_enabled is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_config.py::TestStuckRescueConfig -v`

Expected: FAIL。`test_env_config_defaults` 报 `AttributeError: 'EnvConfig' object has no attribute 'stuck_progress_epsilon'`；两个 CLI 测试报 `SystemExit: 2`（`unrecognized arguments`）。

- [ ] **Step 3a: 加 `EnvConfig` 字段**

在 `src/mad_mario/config.py` 的 `stuck_penalty: float = 5.0` 之后插入：

```python
    # 窗口内净推进阈值（像素）：窗口内最大 x 相对窗口起点推进不足此值即视为未取得进展
    stuck_progress_epsilon: float = 8.0
    # 卡住后随机接管动作尝试脱困；关闭则回到"达到阈值直接判负"的旧行为
    stuck_rescue_enabled: bool = True
    # 救援预算（步）：用尽仍无进展时，按 stuck_penalty_enabled 决定判负或无缝重开
    stuck_rescue_max_steps: int = 120
```

- [ ] **Step 3b: 加 CLI 选项并收窄 `--no-stuck-penalty` 的表述**

把 `src/mad_mario/config.py` 中现有的：

```python
    rw.add_argument("--no-stuck-penalty", action="store_true", help="关闭卡住惩罚")
    rw.add_argument("--stuck-max-steps", type=int, default=EnvConfig.stuck_max_steps,
                    help="连续多少步无位移后判定卡住")
    rw.add_argument("--stuck-penalty", type=float, default=EnvConfig.stuck_penalty,
                    help="卡住时扣除的奖励")
```

替换为：

```python
    rw.add_argument("--no-stuck-penalty", action="store_true",
                    help="关闭卡住惩罚与提前结束（不关脱困；要完全关闭卡住处理需同时传 --no-stuck-rescue）")
    rw.add_argument("--stuck-max-steps", type=int, default=EnvConfig.stuck_max_steps,
                    help="连续多少步无净推进后判定卡住")
    rw.add_argument("--stuck-penalty", type=float, default=EnvConfig.stuck_penalty,
                    help="卡住时扣除的奖励")
    rw.add_argument("--no-stuck-rescue", action="store_true",
                    help="关闭卡住后的随机脱困干预（关闭后恢复为达到阈值直接判负）")
    rw.add_argument("--stuck-rescue-max-steps", type=int, default=EnvConfig.stuck_rescue_max_steps,
                    help="卡住后随机脱困的步数预算")
    rw.add_argument("--stuck-progress-epsilon", type=float, default=EnvConfig.stuck_progress_epsilon,
                    help="窗口内净推进阈值（像素），低于此值视为未取得进展")
```

- [ ] **Step 3c: train 映射**

在 `config_from_train_args()` 的 `stuck_penalty=args.stuck_penalty,` 之后插入：

```python
        stuck_progress_epsilon=args.stuck_progress_epsilon,
        stuck_rescue_enabled=not args.no_stuck_rescue,
        stuck_rescue_max_steps=max(1, args.stuck_rescue_max_steps),
```

- [ ] **Step 3d: play 映射**

在 `config_from_play_args()` 的 `stuck_penalty_enabled=False,` 之后插入：

```python
        stuck_rescue_enabled=True,
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_config.py -v`

Expected: PASS（含既有测试，全部通过）。

- [ ] **Step 5: 全量回归**

Run: `uv run pytest -q`

Expected: 全绿。本任务不改变运行期行为，不应有任何测试转红。

- [ ] **Step 6: 提交**

```bash
git add src/mad_mario/config.py tests/test_config.py
git commit -m "feat(config): 新增卡住脱困的配置字段与 CLI 选项"
```

---

### Task 2: 检测口径改造与类改名（`StuckPenalty` → `StuckHandler`）

**Files:**
- Modify: `src/mad_mario/env/wrappers.py`（`StuckPenalty`，6-65 行）
- Modify: `src/mad_mario/env/factory.py`（import 第 10 行；构造 44-49 行）
- Test: `tests/test_wrappers.py`（import 第 10 行；`TestStuckPenalty` 41-123 行）

**Interfaces:**
- Consumes: Task 1 的 `EnvConfig.stuck_progress_epsilon`
- Produces: `StuckHandler(env, max_stuck_steps, penalty, progress_epsilon=8.0, penalty_enabled=True)`；`info["max_x_pos"]`、`info["stuck_steps"]` 语义保持；`info["stuck"]` 在达到阈值判负时为 `True`

**本任务实现检测口径，不含救援**：达到 `max_stuck_steps` 且 `penalty_enabled=True` 时仍是"扣分 + 截断"。救援状态机在 Task 3 加入，届时 Task 3 会再次修改本任务的 `test_stuck_triggers` 与 `test_penalty_disabled_does_not_truncate`。

`StuckPenalty` 全仓库仅被 `src/mad_mario/env/factory.py:10,45` 与 `tests/test_wrappers.py:10` 引用（已 grep 核实），改名无需保留别名。

- [ ] **Step 1: 改写测试**

编辑 `tests/test_wrappers.py`：

1. 第 10 行的 import 改为：

```python
from mad_mario.env.wrappers import FlagReward, ProgressReward, ScoreReward, StuckHandler
```

2. 把 `class TestStuckPenalty:` 改名为 `class TestStuckHandler:`，并把该类内所有 `StuckPenalty(` 替换为 `StuckHandler(`。

3. 在模块级 `_BaseMockEnv` 类定义之后、`TestStuckHandler` 之前，插入两个**共享** mock 环境。多个测试需要同一种"静止/可推进"环境，共享定义可避免同一段 mock 代码重复五份：

```python
class _StillEnv(_BaseMockEnv):
    """完全静止的环境：x/y 恒定，用于验证卡住判定。"""

    def _make_info(self):
        return {"x_pos": 100.0, "y_pos": 0.0}


class _MovableEnv(_BaseMockEnv):
    """x 可通过 ``env.unwrapped._pos`` 手动推进的环境。"""

    def __init__(self):
        super().__init__()
        self._pos = 100.0

    def _make_info(self):
        return {"x_pos": self._pos, "y_pos": 0.0}
```

4. 把该类内的 4 个既有测试整体替换为下面 10 个（`ProgressReward` / `ScoreReward` / `FlagReward` 三个测试类保持不动）：

```python
class TestStuckHandler:
    def test_not_stuck_moves(self):
        """正常向右推进时不应触发卡住。"""
        class _MovingEnv(_BaseMockEnv):
            def _make_info(self):
                return {"x_pos": float(self._step_n * 10), "y_pos": 0.0}

        env = StuckHandler(_MovingEnv(), max_stuck_steps=3, penalty=5.0,
                           progress_epsilon=8.0)
        obs, info = env.reset()
        assert info["stuck_steps"] == 0

        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not truncated, "推进中不应被截断"
            assert info["stuck_steps"] == 0

    def test_wiggle_does_not_reset(self):
        """x 在 ±1 像素抖动时净推进不足阈值，必须仍判定为卡住。

        旧实现只比较相邻两步，抖动会不断清零计数导致永远不触发；
        本测试是该缺陷的回归证据。
        """
        class _WiggleEnv(_BaseMockEnv):
            def _make_info(self):
                return {"x_pos": 100.0 + (self._step_n % 2), "y_pos": 0.0}

        env = StuckHandler(_WiggleEnv(), max_stuck_steps=3, penalty=5.0,
                           progress_epsilon=8.0)
        env.reset()

        for i in range(2):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not truncated, f"第 {i + 1} 步不应截断"
            assert info["stuck_steps"] == i + 1, "抖动必须累加停滞计数"

        obs, reward, terminated, truncated, info = env.step(0)
        assert truncated, "抖动应被判定为卡住"
        assert reward == -5.0, f"应获得 -5.0 惩罚，实际 {reward}"
        assert info["stuck"] is True

    def test_vertical_only_movement_counts_as_stuck(self):
        """原地反复跳跃（只有 y 变化、x 不推进）同样属于卡住。"""
        class _JumpInPlaceEnv(_BaseMockEnv):
            def _make_info(self):
                return {"x_pos": 100.0, "y_pos": float(self._step_n % 2)}

        env = StuckHandler(_JumpInPlaceEnv(), max_stuck_steps=2, penalty=5.0)
        env.reset()

        for _ in range(2):
            obs, reward, terminated, truncated, info = env.step(0)

        assert truncated, "只有 y 变化必须判定为卡住"

    def test_stuck_triggers(self):
        """站在原地不动超过阈值应惩罚并截断。"""
        env = StuckHandler(_MovableEnv(), max_stuck_steps=3, penalty=5.0)
        env.reset()

        # 前 2 步不动 → 第 3 步触发（>= max_stuck_steps）
        for i in range(2):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not truncated, f"第 {i + 1} 步不应截断"
            assert info["stuck_steps"] == i + 1

        obs, reward, terminated, truncated, info = env.step(0)
        assert truncated, "应被截断"
        assert reward == -5.0, f"应获得 -5.0 惩罚，实际 {reward}"
        assert info["stuck"] is True

    def test_stuck_recovers(self):
        """取得净推进后停滞计数应清零。"""
        env = StuckHandler(_MovableEnv(), max_stuck_steps=3, penalty=5.0,
                           progress_epsilon=8.0)
        env.reset()

        for _ in range(2):  # 卡 2 步
            env.step(0)

        env.unwrapped._pos = 120.0  # 推进 20 >= epsilon
        obs, reward, terminated, truncated, info = env.step(0)
        assert not truncated, "推进后不应截断"
        assert info["stuck_steps"] == 0, "停滞计数应重置"

    def test_small_drift_does_not_reset(self):
        """单步位移小于 progress_epsilon 的累计漂移不应被当作取得进展。"""
        env = StuckHandler(_MovableEnv(), max_stuck_steps=3, penalty=5.0,
                           progress_epsilon=8.0)
        env.reset()

        # 每步只前进 1 像素，累计 3 步仍是 3 < 8
        for i in range(3):
            env.unwrapped._pos += 1.0
            obs, reward, terminated, truncated, info = env.step(0)
            assert info["stuck_steps"] == i + 1, f"第 {i + 1} 步应仍在停滞窗口内"

        assert truncated, "累计推进不足阈值时仍应判定为卡住"

    def test_penalty_disabled_does_not_truncate(self):
        """penalty_enabled=False 时只统计，不扣分不截断。"""
        env = StuckHandler(_StillEnv(), max_stuck_steps=2, penalty=5.0,
                           penalty_enabled=False)
        env.reset()

        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not truncated, "惩罚关闭时不应截断"
            assert reward == 0.0, "惩罚关闭时不应扣分"

        assert info["stuck_steps"] == 5

    def test_flag_get_suppresses_stuck(self):
        """拉旗动画期间 x 可能不增，不应判定为卡住。

        flag_get 抑制的是判负与救援进入，停滞计数仍照常维护。
        """
        class _FlagEnv(_BaseMockEnv):
            def _make_info(self):
                return {"x_pos": 100.0, "y_pos": 0.0, "flag_get": True}

        env = StuckHandler(_FlagEnv(), max_stuck_steps=2, penalty=5.0)
        env.reset()

        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not truncated, "flag_get 为真时不应判负"

        assert info["stuck_steps"] == 5, "计数照常维护，只是不触发判负"

    def test_missing_x_pos_never_triggers(self):
        """info 无 x_pos 时不做判定、不累加计数。"""
        class _NoPosEnv(_BaseMockEnv):
            def _make_info(self):
                return {}

        env = StuckHandler(_NoPosEnv(), max_stuck_steps=2, penalty=5.0)
        env.reset()

        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not truncated, "缺位置信息时不应误判"
            assert info["stuck_steps"] == 0

    def test_stuck_tracks_max_x(self):
        """应追踪历史最大 x，回退时不变小。"""
        env = StuckHandler(_MovableEnv(), max_stuck_steps=10, penalty=5.0)
        obs, info = env.reset()
        assert info["max_x_pos"] == 100.0

        env.unwrapped._pos = 150.0
        obs, reward, terminated, truncated, info = env.step(0)
        assert info["max_x_pos"] == 150.0

        env.unwrapped._pos = 130.0  # 后退
        obs, reward, terminated, truncated, info = env.step(0)
        assert info["max_x_pos"] == 150.0, "max_x_pos 不应缩小"
```

注意 `test_penalty_disabled_does_not_truncate` 只传 `penalty_enabled=False`：本任务还没有 `rescue_enabled` 形参，Task 3 才补上第二个开关（见 Task 3 Step 1 的改写条目）。

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_wrappers.py -v`

Expected: FAIL。`ImportError: cannot import name 'StuckHandler'`（其余测试类也会因 import 失败而报错）；更关键的是 `test_wiggle_does_not_reset` 与 `test_small_drift_does_not_reset` 在旧实现下必然失败——它们是本次修复的核心证据。

- [ ] **Step 3: 改写 `StuckHandler`**

把 `src/mad_mario/env/wrappers.py` 中整个 `StuckPenalty` 类（第 6-65 行，含 `_position_value`）替换为：

```python
class StuckHandler(gym.Wrapper):
    """检测"水平方向长时间无净推进"，并扣分并提前结束回合。

    判定口径：维护一个停滞窗口，窗口内只增不减的最大 x 相对窗口起点推进
    不足 ``progress_epsilon`` 时累加 ``stuck_steps``。只看 ``x_pos``，不看
    ``y_pos``——原地反复跳跃而 x 不前进同样属于卡住。

    ``penalty_enabled=False`` 时只统计，不扣分也不截断。
    """

    def __init__(self, env, max_stuck_steps, penalty, progress_epsilon=8.0,
                 penalty_enabled=True):
        super().__init__(env)
        self.max_stuck_steps = int(max_stuck_steps)
        self.penalty = float(penalty)
        self.progress_epsilon = float(progress_epsilon)
        self.penalty_enabled = bool(penalty_enabled)
        self.best_x = 0.0
        self.window_start_x = 0.0
        self.window_best_x = 0.0
        self.stuck_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        x_pos = self._position_value(info.get("x_pos"))
        self.best_x = x_pos if x_pos is not None else 0.0
        self.window_start_x = self.best_x
        self.window_best_x = self.best_x
        self.stuck_steps = 0
        info["max_x_pos"] = self.best_x
        info["stuck_steps"] = self.stuck_steps
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        x_pos = self._position_value(info.get("x_pos"))

        if x_pos is not None:
            self.best_x = max(self.best_x, x_pos)
            self.window_best_x = max(self.window_best_x, x_pos)
            if self.window_best_x - self.window_start_x >= self.progress_epsilon:
                # 取得进展：窗口基准前移，停滞计数清零
                self.window_start_x = self.window_best_x
                self.stuck_steps = 0
            else:
                self.stuck_steps += 1

        if (
            self.stuck_steps >= self.max_stuck_steps
            and not terminated
            and not truncated
            and not info.get("flag_get")
            and self.penalty_enabled
        ):
            reward = float(reward) - self.penalty
            truncated = True
            info["stuck"] = True

        info["max_x_pos"] = self.best_x
        info["stuck_steps"] = self.stuck_steps
        return obs, reward, terminated, truncated, info

    def _position_value(self, value):
        if value is None:
            return None
        return float(value)
```

关键点（实现者必须理解，不要"优化"掉）：

- 位置判定整体嵌在 `if x_pos is not None:` 内——`x` 缺失时**既不加计数也不判负**。
- `window_start_x` 只在跨过 `progress_epsilon` 时才前移，因此窗口内 `window_best_x` 的上界被钉死在 `window_start_x + progress_epsilon` 以内，`100 ↔ 101` 抖动永远无法伪装成进展。
- 删除了旧的 `movement_epsilon`、`last_x_pos`、`last_y_pos`（新口径不再有"相邻两步位移"这一概念），`progress_epsilon` 取代它。

- [ ] **Step 4: 接线 `factory.py`**

编辑 `src/mad_mario/env/factory.py`：

1. 第 10 行 import 改为：

```python
from mad_mario.env.wrappers import FlagReward, ProgressReward, ScoreReward, StuckHandler
```

2. 第 44-49 行的构造改为：

```python
    if config.stuck_penalty_enabled:
        env = StuckHandler(
            env,
            max_stuck_steps=config.stuck_max_steps,
            penalty=config.stuck_penalty,
            progress_epsilon=config.stuck_progress_epsilon,
            penalty_enabled=config.stuck_penalty_enabled,
        )
```

（注入条件本任务保持 `if config.stuck_penalty_enabled:` 不变，Task 3 再改为析取。）

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_wrappers.py -v`

Expected: PASS，全部 10 个 `TestStuckHandler` 测试通过。

- [ ] **Step 6: 全量回归**

Run: `uv run pytest -q`

Expected: 全绿。若 `tests/test_config.py` 有测试因导入 `StuckPenalty` 而失败（预期没有，已 grep 核实），说明改名遗漏，需一并修复。

- [ ] **Step 7: 提交**

```bash
git add src/mad_mario/env/wrappers.py src/mad_mario/env/factory.py tests/test_wrappers.py
git commit -m "fix(env): 卡住判定改用窗口内 x 净推进，修复贴墙抖动漏判"
```

---

### Task 3: 救援状态机（随机接管动作脱困）

**Files:**
- Modify: `src/mad_mario/env/wrappers.py`（`StuckHandler`）
- Modify: `src/mad_mario/env/factory.py`（注入条件与构造参数）
- Test: `tests/test_wrappers.py`（`TestStuckHandler`）

**Interfaces:**
- Consumes: Task 1 的 `EnvConfig.stuck_rescue_enabled` / `stuck_rescue_max_steps`；Task 2 的 `StuckHandler`；Task 2 在 `tests/test_wrappers.py` 模块级留下的共享 mock `_StillEnv` 与 `_MovableEnv`（本任务直接复用，不要再定义局部副本）
- Produces: `StuckHandler(env, max_stuck_steps, penalty, progress_epsilon=8.0, penalty_enabled=True, rescue_enabled=True, rescue_max_steps=120)`；`info["rescuing"]: bool`、`info["rescue_steps"]: int`

- [ ] **Step 1: 写失败测试**

在 `tests/test_wrappers.py` 的 `TestStuckHandler` 内追加以下 5 个测试，并按说明改写 Task 2 留下的 2 个测试。

新增：

```python
    def test_stuck_enters_rescue(self):
        """达到停滞阈值应进入脱困阶段，而非直接截断。"""
        env = StuckHandler(_StillEnv(), max_stuck_steps=3, penalty=5.0)
        env.reset()

        for i in range(2):
            obs, reward, terminated, truncated, info = env.step(0)
            assert info["rescuing"] is False, f"第 {i + 1} 步不应进入脱困"

        obs, reward, terminated, truncated, info = env.step(0)
        assert info["rescuing"] is True, "达到阈值应进入脱困"
        assert info["rescue_steps"] == 0
        assert not truncated, "进入脱困不应截断"
        assert reward == 0.0, "进入脱困不应扣分"
        assert "stuck" not in info, "进入脱困不置位 stuck 标记"

    def test_rescue_takes_over_action(self, monkeypatch):
        """脱困阶段下发给内层环境的动作来自 action_space，而非调用方传入值。"""
        received = []

        class _RecordingEnv(_BaseMockEnv):
            def _make_info(self):
                return {"x_pos": 100.0, "y_pos": 0.0}

            def step(self, action):
                received.append(action)
                return super().step(action)

        env = StuckHandler(_RecordingEnv(), max_stuck_steps=2, penalty=5.0,
                           rescue_max_steps=5)
        # 固定采样结果，避免概率性断言
        monkeypatch.setattr(env.action_space, "sample", lambda: 3)
        env.reset()

        env.step(0)  # 第 1 步：策略动作
        env.step(0)  # 第 2 步：达到阈值并进入脱困，本步仍是策略动作
        env.step(0)  # 第 3 步：脱困接管

        assert received == [0, 0, 3], f"脱困后应下发采样动作，实际 {received}"

    def test_rescue_exits_on_progress(self):
        """脱困期间取得净推进应立即交还控制权。"""
        env = StuckHandler(_MovableEnv(), max_stuck_steps=2, penalty=5.0,
                           progress_epsilon=8.0, rescue_max_steps=10)
        env.reset()

        env.step(0)  # 停滞 1 步
        env.step(0)  # 达到阈值 → 进入脱困
        env.unwrapped._pos = 120.0  # 脱困中推进 20 >= epsilon

        obs, reward, terminated, truncated, info = env.step(0)
        assert info["rescuing"] is False, "取得推进后应结束脱困"
        assert info["rescue_steps"] == 0
        assert info["stuck_steps"] == 0
        assert not truncated
        assert reward == 0.0

    def test_rescue_budget_exhausted_truncates(self):
        """脱困预算用尽且惩罚开启时应扣分并截断。"""
        env = StuckHandler(_StillEnv(), max_stuck_steps=2, penalty=5.0,
                           rescue_max_steps=3)
        env.reset()

        env.step(0)  # 停滞 1 步
        env.step(0)  # 达到阈值 → 进入脱困

        for i in range(2):  # 脱困第 1、2 步
            obs, reward, terminated, truncated, info = env.step(0)
            assert not truncated, f"脱困第 {i + 1} 步不应截断"
        assert info["rescue_steps"] == 2

        obs, reward, terminated, truncated, info = env.step(0)  # 脱困第 3 步，预算用尽
        assert truncated, "预算用尽应截断"
        assert reward == -5.0
        assert info["stuck"] is True
        assert info["rescuing"] is False

    def test_rescue_budget_exhausted_without_penalty_restarts(self):
        """惩罚关闭时预算用尽不截断，无缝开始新一轮脱困。"""
        env = StuckHandler(_StillEnv(), max_stuck_steps=2, penalty=5.0,
                           penalty_enabled=False, rescue_max_steps=3)
        env.reset()

        env.step(0)
        env.step(0)  # 进入脱困

        for _ in range(3):  # 脱困第 1、2、3 步
            obs, reward, terminated, truncated, info = env.step(0)
            assert not truncated, "惩罚关闭时不应截断"
            assert reward == 0.0, "惩罚关闭时不应扣分"

        assert info["rescuing"] is True, "应保持脱困状态（连续随机挣扎）"
        assert info["rescue_steps"] == 0, "预算应已原地重置"

    def test_no_rescue_truncates_at_threshold(self):
        """rescue_enabled=False 时保持旧行为：达到阈值直接扣分截断。"""
        env = StuckHandler(_StillEnv(), max_stuck_steps=2, penalty=5.0,
                           rescue_enabled=False)
        env.reset()

        env.step(0)
        obs, reward, terminated, truncated, info = env.step(0)
        assert truncated, "关闭脱困后应恢复为直接判负"
        assert reward == -5.0
        assert info["stuck"] is True
```

改写 Task 2 留下的测试：

1. `test_stuck_triggers`：新语义下达到阈值只是**进入脱困**，不再截断。**整体删除**该测试（其职责已被 `test_stuck_enters_rescue` 承接，判负路径由 `test_no_rescue_truncates_at_threshold` 与 `test_rescue_budget_exhausted_truncates` 覆盖）。

2. `test_penalty_disabled_does_not_truncate`：把构造改为显式关闭脱困，使该测试只考察"惩罚开关"，不被脱困分支干扰：

```python
        env = StuckHandler(_StillEnv(), max_stuck_steps=2, penalty=5.0,
                           penalty_enabled=False, rescue_enabled=False)
```

3. 同理，下列三个**检测口径**测试的意图是验证"卡住能被正确判出"，一旦默认开启脱困，达到阈值就不再截断，断言必然失败。它们同样需要补 `rescue_enabled=False`，把考察范围收回到检测本身——与第 2 条同一模式：

```python
# test_wiggle_does_not_reset / test_small_drift_does_not_reset / test_vertical_only_movement_counts_as_stuck
        env = StuckHandler(_MovableEnv(), max_stuck_steps=..., penalty=5.0,
                           progress_epsilon=8.0, rescue_enabled=False)
```

改完后 `TestStuckHandler` 共 15 个测试（Task 2 的 10 个 − 删除 1 个 + 新增 6 个）。

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_wrappers.py -v`

Expected: FAIL。新测试报 `TypeError: __init__() got an unexpected keyword argument 'rescue_enabled'`（`rescue_max_steps` 同理）；`info["rescuing"]` 触发 `KeyError`。

- [ ] **Step 3: 加入状态机**

把 `src/mad_mario/env/wrappers.py` 的 `StuckHandler` 整体替换为下述最终版本（并把类 docstring 换成这版）：

```python
class StuckHandler(gym.Wrapper):
    """检测"水平方向长时间无净推进"，先随机接管动作尝试脱困，再兜底判负。

    判定口径：维护一个停滞窗口，窗口内只增不减的最大 x 相对窗口起点推进
    不足 ``progress_epsilon`` 时累加 ``stuck_steps``。只看 ``x_pos``，不看
    ``y_pos``——原地反复跳跃而 x 不前进同样属于卡住。

    状态机：

    - NORMAL：停滞满 ``max_stuck_steps`` 且 ``rescue_enabled`` 时进入 RESCUE；
      未开启脱困时按 ``penalty_enabled`` 直接判负（旧行为）。
    - RESCUE：忽略传入动作，改用 ``action_space.sample()`` 随机尝试；取得净
      推进即回到 NORMAL。
    - 救援满 ``rescue_max_steps`` 仍无进展时：``penalty_enabled`` 为真则扣分
      并截断，否则原地开始新一轮救援（播放模式：永不因卡住而结束关卡）。
    """

    def __init__(self, env, max_stuck_steps, penalty, progress_epsilon=8.0,
                 penalty_enabled=True, rescue_enabled=True, rescue_max_steps=120):
        super().__init__(env)
        self.max_stuck_steps = int(max_stuck_steps)
        self.penalty = float(penalty)
        self.progress_epsilon = float(progress_epsilon)
        self.penalty_enabled = bool(penalty_enabled)
        self.rescue_enabled = bool(rescue_enabled)
        self.rescue_max_steps = int(rescue_max_steps)
        self.best_x = 0.0
        self.window_start_x = 0.0
        self.window_best_x = 0.0
        self.stuck_steps = 0
        self.rescue_steps = 0
        self.rescuing = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        x_pos = self._position_value(info.get("x_pos"))
        self.best_x = x_pos if x_pos is not None else 0.0
        self.window_start_x = self.best_x
        self.window_best_x = self.best_x
        self.stuck_steps = 0
        self.rescue_steps = 0
        self.rescuing = False
        info["max_x_pos"] = self.best_x
        info["stuck_steps"] = self.stuck_steps
        info["rescuing"] = self.rescuing
        info["rescue_steps"] = self.rescue_steps
        return obs, info

    def step(self, action):
        if self.rescuing:
            # 脱困阶段接管动作：确定性策略会被同一障碍反复挡住，只有随机采样才能打破死循环
            action = self.action_space.sample()

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        x_pos = self._position_value(info.get("x_pos"))

        if x_pos is not None:
            self.best_x = max(self.best_x, x_pos)
            self.window_best_x = max(self.window_best_x, x_pos)
            if self.window_best_x - self.window_start_x >= self.progress_epsilon:
                # 取得进展：窗口基准前移、停滞计数清零、结束脱困
                self.window_start_x = self.window_best_x
                self.stuck_steps = 0
                self.rescuing = False
                self.rescue_steps = 0
            elif self.rescuing:
                self.rescue_steps += 1
            else:
                self.stuck_steps += 1

        if not terminated and not truncated and not info.get("flag_get"):
            if not self.rescuing and self.stuck_steps >= self.max_stuck_steps:
                if self.rescue_enabled:
                    # 当步判定之后置位，因此下一步的动作才被接管
                    self.rescuing = True
                    self.rescue_steps = 0
                elif self.penalty_enabled:
                    reward = float(reward) - self.penalty
                    truncated = True
                    info["stuck"] = True
            elif self.rescuing and self.rescue_steps >= self.rescue_max_steps:
                if self.penalty_enabled:
                    self.rescuing = False
                    reward = float(reward) - self.penalty
                    truncated = True
                    info["stuck"] = True
                else:
                    # 播放语义：不判负，原地开始新一轮随机尝试（不留策略动作的空隙）
                    self.rescue_steps = 0

        info["max_x_pos"] = self.best_x
        info["stuck_steps"] = self.stuck_steps
        info["rescuing"] = self.rescuing
        info["rescue_steps"] = self.rescue_steps
        return obs, reward, terminated, truncated, info

    def _position_value(self, value):
        if value is None:
            return None
        return float(value)
```

关键点（不要"优化"掉）：

- 动作接管必须在 `self.env.step(action)` **之前**，否则该步动作已经下发。
- 进入脱困当步的动作**仍是策略动作**（判定发生在 `step` 之后），接管从下一步开始——`test_rescue_takes_over_action` 的 `[0, 0, 3]` 断言固化了这一点。
- `elif self.rescuing: self.rescue_steps += 1` 与 `else: self.stuck_steps += 1` 互斥：脱困期间 `stuck_steps` 冻结在阈值上（保留"触发救援时的停滞步数"这一诊断量）。
- 救援失败且 `penalty_enabled=False` 时**只**重置 `rescue_steps`，保持 `rescuing=True`——若顺手把 `rescuing` 置假，下一步会因 `stuck_steps` 仍等于阈值而立刻重入脱困，中间插入一步策略动作，语义含混。

- [ ] **Step 4: 接线 `factory.py`**

编辑 `src/mad_mario/env/factory.py`，把第 44 行起的注入改为：

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

两个开关全为假时不注入包装器，行为与改动前一致（零开销）。

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_wrappers.py -v`

Expected: PASS。

- [ ] **Step 6: 全量回归**

Run: `uv run pytest -q`

Expected: 全绿。

- [ ] **Step 7: 端到端冒烟（可选但推荐）**

用极短训练确认环境链能正常构造、包装器不炸：

```bash
uv run python -m mad_mario train --timesteps 2048 --num-envs 2 --flat-save-root --save-root /tmp/madmario_smoke
```

Expected: 进程正常结束，打印"训练完成。"，`/tmp/madmario_smoke/mario_dqn_final.zip` 存在。随后清理：`rm -rf /tmp/madmario_smoke`。

若该命令因需要图形/显示而失败，记为环境限制、跳过即可，不要为此修改代码。

- [ ] **Step 8: 提交**

```bash
git add src/mad_mario/env/wrappers.py src/mad_mario/env/factory.py tests/test_wrappers.py
git commit -m "feat(env): 卡住后随机接管动作脱困，播放模式不再永久站住"
```

---

## 完成后验收

- [ ] `uv run pytest -q` 全绿。
- [ ] `uv run python -m mad_mario train --help` 输出含 `--no-stuck-rescue`、`--stuck-rescue-max-steps`、`--stuck-progress-epsilon`。
- [ ] 矩阵四行均有测试覆盖：第 1 行 `test_rescue_budget_exhausted_truncates`、第 2 行 `test_rescue_budget_exhausted_without_penalty_restarts`、第 3 行 `test_no_rescue_truncates_at_threshold`、第 4 行由 `factory.py` 的注入条件保证（两开关全假时不注入）。
- [ ] `x` 抖动不再漏判：`test_wiggle_does_not_reset` 通过（旧实现下必然失败）。
- [ ] `info["stuck"]` 只在判负时为 `True`：`test_stuck_enters_rescue` 断言 `"stuck" not in info`。
