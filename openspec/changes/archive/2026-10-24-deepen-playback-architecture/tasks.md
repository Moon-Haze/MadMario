# Tasks

## 1. 播放环境语义单点化（design D1）

- [x] 1.1 在 `config.py` 新增纯函数 `playback_env_config(env_config) -> EnvConfig`（关闭 clip/progress/score/flag/stuck_penalty，保留 stuck_rescue；实施时从 factory 移至 config，避免循环导入，见 design D1 位置修正），并为它写接口级单测：断言 6 个开关的播放取值、其余字段原样保留——对应 spec `playback` 的"播放环境语义"需求
- [x] 1.2 将 `config.py` 的 `config_from_play_args` 与 `best_playback.py` 的 `_to_playback_config` 改为调用该函数，删除两处内联的 6 布尔赋值；验证：`pytest tests/test_config.py tests/test_factory.py tests/test_best_playback.py` 全绿（115 passed），且 `grep -rn "stuck_rescue_enabled=True" src/` 只命中 config.py 一处

## 2. PlaybackSession：播放链路开真接缝（design D2）

- [x] 2.1 记录基线：全量运行 `pytest tests/test_video.py`，确认重构前行为断言（目录树、命名、并发 Barrier、日志文案）全部通过（51 passed in 4.01s）——这些断言是重构期间的行为护栏
- [x] 2.2 在 `playback/runner.py` 内新增 `PlaybackSession`（构造注入 load_model / make_env / open_viewer / thread_count，默认值即现有生产实现），把 `play()` 的编排移入 `run(config)`；`play()` 缩为薄入口并自带 KeyboardInterrupt 消化，`catch_interrupt` 参数从 interface 删除；验证：`play(config)` 仍可无参协作者直接调用（默认协作者生效）
- [x] 2.3 将 `tests/test_video.py` 改为构造注入驱动（PlaybackSession(load_model=fake, make_env=fake_env, open_viewer=...)），删除 8 处模块级 monkeypatch 与对 SB3 numpy 种子行为的复刻；断言全部保留；验证：`pytest tests/test_video.py` 全绿且文件行数明显回落（目标 < 700 行）
- [x] 2.4 补中断语义接口级测试：`run()` 不吞 KeyboardInterrupt（训练内嵌可冒泡）、`play()` 薄入口吞掉并记日志——对应 spec `playback` 的"播放中断"需求
- [x] 2.5 更新 README 中 play 相关描述（如涉及协作者行为变化）；验证：README 示例命令与 `python -m mad_mario.cli play --help` 一致

## 3. 最优录制协议收拢（design D3，依赖组 1、2）

- [x] 3.1 从 `BestModelPlaybackCallback` 拆出 `BestRunRecorder`（interface：enqueue(step, checkpoint, video_dir) / drain(timeout)），快照复制保留在 enqueue 前的主线程路径；worker 改用 `PlaybackSession.run`；为 recorder 写接口级单测（排队、单趟失败继续、drain 幂等）——对应 spec `best-run-recording` 的"后台录制不阻塞训练"与"收尾排水保证"需求
- [x] 3.2 改造 `trainer.py`：在训练配置上经 `playback_env_config` 生成播放配置并构造 recorder；回调构造不再透传 `train_config` 整树；排水只保留 `trainer.finally` 一处，删除 `_on_training_end` 里的排水；验证：`grep -n "wait_for_recording\|drain" src/mad_mario/training/*.py` 只剩 trainer 一处调用
- [x] 3.3 改写 `tests/test_best_playback.py`：门槛判定、快照冻结、排水单点的行为断言改为通过回调与 recorder 的公开 interface 驱动；验证：`pytest tests/test_best_playback.py tests/test_trainer.py` 全绿，mock 桩数量较改写前下降（目标 < 60 处）
- [x] 3.4 训练侧集成冒烟：以极小步数（total_timesteps ≤ 200、eval_freq 调小）跑一次 `train()`，验证 best_model 存档与 `best_model_videos/step_*` 目录按 spec"录制产物按步数分档"落盘、进程退出后 mp4 可打开（文件大小 > 0 且非半截索引）

## 4. config 拆分与假旋钮删除（design D4）

- [x] 4.1 新建 `src/mad_mario/artifacts.py`，迁入 `resolve_checkpoint` 与 `compatible_save_root`，`config.py` / `trainer.py` / `cli.py` 的导入改指新模块并在 config.py 保留转发导入一个过渡提交后移除；验证：`pytest tests/test_config.py tests/test_trainer.py` 全绿
- [x] 4.2 对照验证 render_mode 默认行为：写测试确认 `make_mario_env` 不显式传 `render_mode` 时（训练 thunk 路径）环境构造与 `reset/step` 正常且无渲染副作用，与现状一致——这是删除 `EnvConfig.render_mode` 的安全前提
- [x] 4.3 删除 `EnvConfig.state_dim`（零读取）、`EnvConfig.render_mode` 字段与 `--render-mode` CLI 选项，`make_mario_env` 的 `render_mode` 形参保留；同步更新 README（BREAKING：--render-mode 不再接受）；验证：`pytest` 全量绿、`python -m mad_mario.cli play --help` 输出无 `--render-mode`、`grep -rn "state_dim\|render_mode" src/ README.md` 无残留引用
- [x] 4.4 迁移后收口：确认 `config.py` 不再含磁盘 I/O 策略（`grep -n "glob\|iterdir" src/mad_mario/config.py` 为空），测试覆盖随迁移落在 `tests/test_artifacts.py`

## 5. metrics 清理（design D5）

- [x] 5.1 删除 `logging/metrics.py` 的 `MetricLogger`（约 170 行）及 tutorial 中对它的引用；CSV 表头双份定义收成单一 `CSV_COLUMNS` 常量并两处引用；验证：`pytest tests/test_metrics.py` 全绿、`grep -rn "MetricLogger" src/ tests/ tutorial/` 无残留
- [x] 5.2 matplotlib 后端契约内聚：`metrics.py` 在导入 pyplot 前自行 `matplotlib.use("Agg")`（Tk/GC 跨线程知识随 docstring 迁入），删除 `cli.py` 的 `_use_headless_backend`；验证：`pytest tests/test_cli_logging.py tests/test_metrics.py` 全绿，且 `python -m mad_mario.cli train --help` 路径无 Tk/后端警告

## 6. 集成验证（依赖组 1-5）

- [x] 6.1 全量 `pytest` 通过；对照 `specs/playback/spec.md` 与 `specs/best-run-recording/spec.md` 逐条场景核验：每个场景至少有一个对应测试或冒烟验证，缺者补齐
- [x] 6.2 端到端冒烟：小规模 `train`（Ctrl+C 中断一次验证排水日志"等待后台录制完成"后干净退出）+ `play --sample 2`（验证探索率子目录与视频落盘），与 spec 场景一致

## Workflow follow-up

- 实施由 `/openspec-apply-change` 按本清单执行，逐组提交。
- 全部任务完成并复审后，用 `/openspec-archive-change` 归档，将两份 spec delta 落为主规格。
