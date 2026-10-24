# Proposal

## Why

架构评审发现播放链路与配置模块存在系统性浅模块问题：`play()` 一次协调 14 件事但唯一注入接缝是模块全局名，导致 `tests/test_video.py` 1035 行中约 450 行是假件基建加 8 处 monkeypatch，且假件复刻了 3 条 SB3/matplotlib 内部行为；"播放栈"语义以 6 个布尔在 `config.py` 与 `best_playback.py` 两处逐字重复；最优模型录制的排水协议在 trainer 与回调里各写一遍；`config.py` 664 行承载四种职责并含有死字段与失效旋钮；`logging/metrics.py` 尾部压着 170 行零调用点死类。深化这些模块把复杂度收进 implementation，让测试通过 interface 传假件，测试规模随接缝深化自然回落。

## What Changes

- **播放栈落名**：`env/factory.py` 新增命名的播放环境入口（如 `make_play_env()`），"裁剪/塑形/惩罚关、脱困留"的播放语义单点定义，消除 `config.py:639` 与 `best_playback.py:43` 的 6 布尔逐字重复。
- **播放链路开真接缝**：`play()` 的编排收进构造注入的会话模块（如 `PlaybackSession`）——模型加载、环境工厂、viewer、执行器成为构造参数；计划/单趟执行/串行并发派发收进其 implementation；测试在构造边界注入假件，替代 8 处模块级 monkeypatch。
- **录制协议收拢**：最优模型录制拆出显式录制器 interface（enqueue + drain），排水协议单点化；去掉 `catch_interrupt` 调用方分类与整棵 `train_config` 透传。
- **config 拆分**：`resolve_checkpoint`、`compatible_save_root`、`load_config_file` 迁入独立模块；删除死字段 `EnvConfig.state_dim` 与失效选项 `--render-mode`（**BREAKING**：CLI 不再接受该选项；行为无变化，它本就被 runner 无条件覆盖）。
- **metrics 清理**：删除零调用点的 `MetricLogger`（170 行）；CSV 表头收成单一常量；pyplot 后端契约收进模块自身，`cli.py` 不再背负导入顺序责任。

## Capabilities

### New Capabilities
- `playback`：play 命令的播放执行能力——贪心/采样播放、视频落盘与命名、并发播放、GUI 降级与中断策略。本变更替换其实现（会话模块），行为规格保持不变并被首次钉住。
- `best-run-recording`：训练中达标最优模型的视频录制协议——门槛判定、后台线程录制、收尾排水保证。排水协议从双写收紧为单点。

### Modified Capabilities

（无——项目当前没有任何 specs，`openspec/` 下为空白起点。）

## Impact

- **代码**：`src/mad_mario/playback/`（runner/viewer/video）、`src/mad_mario/training/`（trainer/best_playback）、`src/mad_mario/config.py`、`src/mad_mario/logging/metrics.py`、`src/mad_mario/cli.py`、`src/mad_mario/env/factory.py`。
- **测试**：`tests/test_video.py`（预期显著缩减）、`tests/test_best_playback.py`（83 处 mock 随接缝深化下降）、`tests/test_config.py`、`tests/test_factory.py`、`tests/test_metrics.py` 相应调整。
- **CLI**：删除失效选项 `--render-mode`（行为无变化）；其余命令行行为不变。
- **依赖**：无新增第三方依赖。
