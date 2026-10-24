# Design

## Context

见 proposal.md 的 Why。现状关键事实：

- `play()`（`playback/runner.py:406`）编排 14 件事，协作者（DQN.load、make_mario_env、viewer、ThreadPoolExecutor）只能按模块全局名替换；`tests/test_video.py` 为此维护 8 个 patch 点与约 200 行假件。
- 播放环境语义（裁剪/塑形/惩罚关、脱困留）以 6 个布尔赋值逐字出现在 `config_from_play_args` 与 `best_playback.py:43`（`_to_playback_config`）。
- `BestModelPlaybackCallback` 继承 SB3 `EvalCallback`，同时拥有门槛记账、存档、快照、队列、worker 线程与排水；trainer 在 `finally` 与回调 `_on_training_end` 两处排水，靠 `catch_interrupt` flag 区分调用方，构造时透传整棵 `train_config`。
- `config.py`（664 行）混合 dataclass、argparse、`resolve_checkpoint`、`compatible_save_root`、`load_config_file`；含死字段 `state_dim` 与被 `runner.py` 无条件覆盖的 `render_mode` / `--render-mode`。
- `logging/metrics.py` 尾部 170 行 `MetricLogger` 零调用点；CSV 表头双份定义；matplotlib Agg 后端契约由 `cli.py:_use_headless_backend` 在导入顺序上保证。

行为契约已钉在 `specs/playback/spec.md` 与 `specs/best-run-recording/spec.md`——本变更换实现、不换行为，spec 场景即重构验收基线。

## Goals / Non-Goals

**Goals:**

- 播放链路的协作者全部改为构造期注入，测试在接缝上传真件，删除模块级 monkeypatch。
- "播放环境语义"单点定义，两处调用方指名获取。
- 最优录制的排水协议单点化，接缝上不再传递整棵训练配置。
- config.py 里的 I/O 策略迁出，死字段与假旋钮删除。
- metrics 死代码删除，后端契约内聚到唯一写者。

**Non-Goals:**

- 不改任何训练/播放的可观察行为（spec 场景前后一致）。
- 不把 argparse 定义拆出 config.py——"字段 + flag + 组装"三联画的局部性在同文件内反而最好，拆开会让新增一个旋钮跨两个文件。
- 不给 `train()` 单独立接口级测试目标；它随接缝深化自然变薄。
- 不引入新第三方依赖。

## Decisions

### D1 · 播放环境语义 = config 内的纯函数变换（EnvConfig 的伴生）

`config.py` 新增 `playback_env_config(env_config: EnvConfig) -> EnvConfig`（紧邻 EnvConfig 定义）：返回关闭 `clip_rewards`、`stuck_penalty_enabled`、`progress_reward_enabled`、`score_reward_enabled`、`flag_reward_enabled`，保留 `stuck_rescue_enabled` 的副本（dataclasses.replace）。`config_from_play_args` 与 `_to_playback_config` 都改为调用它。

- **位置修正（实施时发现）**：设计初稿放在 factory，但 `config.py` 导入 factory 会与 factory 既有的 `from mad_mario.config import EnvConfig` 循环导入。它是 EnvConfig 的取值变换（配置层知识），wrapper 组装顺序才是 factory 的知识——两层各守各的，放 config 既避开循环也语义更准。
- **为什么是配置变换而不是 `make_play_env()` 直建环境**：`make_mario_env` 已经是深模块且接口正确（env 进 env 出），缺的只是"播放意图"这个名字。纯函数变换可被两处调用方按需继续组装各自的 `AppConfig`（CLI 播放配置、训练内录制配置结构不同），且纯函数零 mock 即可测。
- **替代方案（否决）**：`make_play_env(config)` 直建环境——会与 `make_mario_env` 形成 sibling 双入口，`render_mode`、`level_name` 等参数面复制一份，浅模块。

### D2 · `PlaybackSession`：play() 的编排收进构造注入的会话模块

`playback/runner.py` 内新增 `PlaybackSession`，`play()` 缩为薄入口：

```
PlaybackSession(                    # interface（窄）
    load_model    = DQN 封装,       # (Path) -> model
    make_env      = make_mario_env 封装（含 rgb_array 强制）,
    open_viewer   = _open_viewer 封装,
    thread_count  = os.cpu_count()//2 封装,
).run(config: AppConfig) -> None    # 串行/并发派发、计划、重播种、中断策略全部收进 implementation
```

- 计划纯函数（`_plan_playthroughs`、`_unique_path`、`_draw_exploration_rate` 等）留在文件内作为 implementation，不对外导出。
- **`catch_interrupt` 从 interface 上删除**：`play()` 薄入口（CLI 专用）自己 try/except KeyboardInterrupt 打日志；`PlaybackSession.run` 永不吞中断。调用方分类从被调方接口消失，训练内 worker 直接 `run()`，中断自然冒泡。
- **测试面**：`tests/test_video.py` 改为构造 `PlaybackSession(load_model=fake, make_env=fake_env, open_viewer=..., ...)`，现有断言（目录树、命名、并发 Barrier、日志文案）保留——它们本来就在测用户可见结果，改的只是驱动方式。
- **替代方案（否决）**：把 session 拆到新文件 `playback/session.py`——计划/命名/派发助手是它的 implementation，分文件只会把一次修改摊到两个文件。

### D3 · `BestRunRecorder` 与评估回调分家，单点排水

`training/best_playback.py` 内拆出 `BestRunRecorder`，`BestModelPlaybackCallback` 缩为"评估回调 + 门槛判定 + 存档 + 委托"：

```
BestRunRecorder(                       # interface（窄）
    session          = 生产 PlaybackSession 工厂,
    playback_config  = 由训练配置经 D1 变换生成的播放 AppConfig（不含训练超参树）,
).enqueue(step, checkpoint, video_dir) -> bool
 .drain(timeout=None)                  # 幂等；快照复制仍在 enqueue 前的主线程完成
```

- **排水只留 trainer 一处**：`trainer.py` 的 `finally` 调 `recorder.drain()`（正常结束与 Ctrl+C 两条路径都过 finally）；`_on_training_end` 里的排水删除。recorder 由 trainer 构造并持有，"谁构造谁收尾"。
- **接缝收窄**：回调构造参数不再透传 `train_config`；`_to_playback_config` 的变换在 trainer 侧完成一次（用 D1），recorder 只收播放配置。checkpoint 每任务自带。
- **保留 EvalCallback 子类化**：门槛判定依赖父类 `best_mean_reward` 的前后比较，外部轮询不可行（现有注释已论证）。
- **替代方案（否决）**：recorder 独立成新文件 `training/recorder.py`——门槛、存档、录制协议是同一个"最优"概念的 implementation，同文件的 locality 最好。

### D4 · config.py：I/O 策略迁出，死字段与假旋钮删除

- `resolve_checkpoint`、`compatible_save_root` 迁至新模块 `src/mad_mario/artifacts.py`（checkpoint 发现 + 目录命名策略）；`config.py` 从 trainer/cli 的导入改指新地址。
- `load_config_file` / `_extract_config_path` / `_apply_config_defaults` **留在 config.py**：它们是 CLI 配置装配的一部分，不是磁盘工件策略。
- 删除 `EnvConfig.state_dim`（全项目零读取）与 `EnvConfig.render_mode` + `--render-mode` 选项（`runner.py` 无条件以 `rgb_array` 覆盖，`make_mario_env` 的 `render_mode` 形参保留——调用方显式传入才是唯一正路）。README 同步。
- **迁移注意**：删 `render_mode` 字段后 `make_mario_env` 的默认 `render_mode=None` 直传 `gym_super_mario_bros.make`——需在任务中验证训练 thunk 路径（不显式传）行为与现状一致（nes 环境默认无渲染，SubprocVecEnv 不消费渲染）。

### D5 · metrics.py：死代码删除，后端契约内聚

- 删除 `MetricLogger`（170 行，零调用点；tutorial 引用一并清理）。
- CSV 表头双份定义收成模块级 `CSV_COLUMNS` 常量。
- `cli.py` 的 `_use_headless_backend` 删除；`metrics.py` 在导入 pyplot 之前自行 `matplotlib.use("Agg")`，Tk/GC 跨线程知识（现 cli docstring）随迁。播放路径不导入 metrics，不受影响——matplotlib 仍可为播放自选后端。

## Risks / Trade-offs

- **[删 `render_mode` 改变 gym 默认渲染]** → 迁移任务里加对照验证：训练 thunk 与评估 thunk 不传 `render_mode` 时，环境构造、步进与现状逐位一致。
- **[PlaybackSession 重构一次动 14 件事]** → 先以现有 `test_video.py` 的行为断言为基线全部跑绿，再动结构；spec 场景（目录树、并发、中断、降级）逐条对应保留的测试。
- **[EvalCallback 父类内部行为（`best_mean_reward` 可变状态、`_on_training_end` 时机）]** → 保留子类化（替代方案已被现有实现论证否决）；D3 只动我们自己拥有的接缝。
- **[matplotlib 后端改为 metrics 自管后，若未来有人在 pyplot 已初始化后导入 metrics]** → `matplotlib.use` 幂等且可后切；训练路径 metrics 是 pyplot 首个导入者，顺序安全；在模块 docstring 记录这条约束。
- **[--render-mode 是 BREAKING]** → README 与 CLI 帮助同步更新；该选项本就无效，无迁移成本。

## Migration Plan

按依赖顺序五组任务，每组完成即全量测试通过、可独立提交：

1. **播放语义单点**（D1）——最小改动，先消除 6 布尔双份。
2. **PlaybackSession**（D2）——runner 重构 + test_video 改为构造注入。
3. **录制器收拢**（D3）——依赖 D1（播放配置变换）与 D2（session interface）。
4. **config 拆分与删除**（D4）——独立，任何时候可做；README 同步。
5. **metrics 清理**（D5）——独立，纯删除加常量收口。

回滚：每组是独立提交，revert 单组即可；无数据迁移、无持久化格式变化。

## Open Questions

无——以上决策均可在现有代码与测试范围内落实。
