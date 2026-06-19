# MadMario 多环境训练重构设计

日期：2026-06-19

## 背景

当前项目已经有单环境训练入口 `main.py` 和多环境训练入口 `main_vector.py`。多环境版本已经能通过 `AsyncVectorEnv` 并行采样，并且 `Mario.act_batch()`、按 transition 数衰减 epsilon、按 step 补齐学习次数等核心语义基本正确。

主要问题是：经验回放直接保存 float32 堆叠状态，长训内存压力很大；多环境训练没有记录真实 loss/Q 曲线；环境数量、训练回合、恢复策略等参数不够灵活；checkpoint 文件名会在同一保存区间内重复覆盖；环境创建逻辑在训练和播放脚本中重复。

本次目标是做一次较彻底但可控的重构，保持 DQN 算法和网络结构不变，优先提升稳定性、可观测性和后续扩展性。

## 目标

1. 降低 replay buffer 内存占用。
2. 修复多环境训练的 loss/Q 日志，使指标可信。
3. 参数化训练入口，尤其是 `episodes`、`num_envs`、checkpoint 恢复策略和保存目录。
4. 改进 checkpoint 命名，保留 `latest.chkpt` 的自动恢复能力，同时避免历史 checkpoint 被频繁覆盖。
5. 抽出环境工厂、训练配置、经验回放和训练循环，让代码职责更清晰。
6. 保持现有 checkpoint 在动作空间一致时仍可加载。

## 非目标

本次不做以下内容：

- 不修改 `MarioNet` 网络结构。
- 不切换到 PPO、A3C 或其他强化学习算法。
- 不实现 Prioritized Replay。
- 不实现跨关卡 curriculum training。
- 不修改动作空间，仍使用 `COMPLEX_MOVEMENT`。
- 不删除旧 checkpoint。

## 模块设计

### `config.py`

新增训练配置模块，定义 `TrainingConfig` 数据类和默认值。配置项包括：

- `episodes`
- `state_dim`
- `replay_capacity`
- `batch_size`
- `burnin`
- `learn_every`
- `sync_every`
- `save_every`
- `num_envs`
- `save_root`
- `checkpoint`
- `resume`

`main.py` 和 `main_vector.py` 负责解析命令行参数，并用这些参数覆盖默认配置。

### `env_factory.py`

新增环境工厂模块，统一创建训练和播放环境。核心函数：

- `make_mario_env(render_mode=None)`：创建单个 Mario 环境并应用 Joypad、SkipFrame、灰度、缩放、归一化、FrameStack。
- `make_env_thunk(render_mode=None)`：返回可被 vector env 使用的 thunk。
- `make_vector_env(num_envs)`：创建 `AsyncVectorEnv`。

这样训练和播放脚本共用同一套预处理链，避免 wrapper 不一致。

### `replay_buffer.py`

新增紧凑经验回放模块。对外接口：

- `push(state, next_state, action, reward, done)`
- `sample(batch_size, device)`
- `__len__()`

内部存储策略：

- 输入状态来自 wrapper，通常是 float32 `[0, 1]`。
- 写入 replay 时转为 uint8 `[0, 255]` 保存。
- sample 时转回 float32 并除以 255，输出给网络。

这种方式将每个状态从 float32 降为 uint8，显著降低 replay buffer 内存压力。为降低改动风险，本次仍保存完整的 `state` 和 `next_state`，不做单帧去重。

### `agent.py`

保留 `Mario` 类作为智能体核心，但调整职责：

- 使用 `ReplayBuffer` 替代内置 `deque`。
- 构造函数接收配置项，避免 batch size、burnin 等超参数散落在类内部。
- 保留 `act()` 和 `act_batch()`。
- `cache()` 和 `cache_batch()` 委托给 replay buffer。
- `recall()` 从 replay buffer 采样。
- 保存 checkpoint 时使用更明确的文件名。

checkpoint 内容继续包含：

- model
- optimizer
- exploration_rate
- curr_step
- episode
- save_dir

### `trainer.py`

新增训练循环模块，提供：

- `train_single_env(config)`
- `train_vector_env(config)`

单环境训练保持原有逻辑；多环境训练负责：

- 批量动作选择。
- 批量环境 step。
- 处理 vector env 的 `final_observation`。
- 批量写入 replay。
- 调用 `mario.learn()`。
- 统计每个环境的 episode reward 和 length。
- 用最近 loss/Q rolling window 写入 episode 指标。
- 定期记录 metrics 并保存 checkpoint。

### `main.py` 和 `main_vector.py`

两个入口文件改为轻量入口：

- 解析命令行参数。
- 构建 `TrainingConfig`。
- 调用对应 trainer。

示例：

```bash
python main.py --episodes 1000 --no-resume
python main_vector.py --num-envs 4 --episodes 40000
```

### `play_trained.py`

播放脚本改用 `env_factory.py` 创建环境，确保播放预处理和训练预处理一致。默认加载文件可以继续保持 `trained_mario.chkpt`，不强制改成 `latest.chkpt`。

## 日志设计

多环境训练中维护两个 rolling window：

- `recent_losses`
- `recent_qs`

每次 `mario.learn()` 返回非空 loss/Q 时追加。每个环境 episode 结束时，调用 `MetricLogger.log_episode_metrics()` 写入：

- episode reward
- episode length
- 最近 loss 均值
- 最近 Q 均值

`MetricLogger.record()`、CSV 和 plot 生成逻辑尽量保持不变。

状态栏中的奖励展示改为明确含义，避免把正在进行中的环境平均奖励误称为“最近奖励”。优先展示最近完成 episode 的 rolling mean。

## Checkpoint 设计

保存文件名改为：

```text
mario_net_step_<curr_step>_ep_<episode>.chkpt
```

同时继续更新：

```text
checkpoints/latest.chkpt
```

恢复逻辑：

- 默认如果 `checkpoints/latest.chkpt` 存在则恢复。
- `--checkpoint <path>` 指定恢复文件。
- `--no-resume` 强制从头训练。

加载旧 checkpoint 时保持兼容，只要 checkpoint 中包含已有字段即可。

## 错误处理和兼容性

- Windows 下保留 `if __name__ == "__main__": main()`，避免 vector env 子进程递归启动。
- vector env 的 thunk 放在可 pickle 的位置，降低 Windows multiprocessing 问题。
- 如果 checkpoint 文件不存在，明确报错或按恢复策略从头开始。
- 如果动作空间与 checkpoint 不匹配，PyTorch 加载会报 shape mismatch，保留真实错误信息。

## 测试计划

1. 语法检查：

```bash
python -m py_compile *.py
```

2. ReplayBuffer 小测试：

- push 随机状态。
- sample 后确认 shape、dtype 和数值范围。
- 确认 action、reward、done dtype 正确。

3. 多环境短训 smoke test：

```bash
python main_vector.py --num-envs 2 --episodes 2 --no-resume
```

验证：

- vector env 能 reset/step。
- metrics.csv 能写入。
- checkpoint 能保存。
- loss/Q 在 burnin 前允许为空或为 0，但记录逻辑不报错。

4. 单环境短训 smoke test：

```bash
python main.py --episodes 2 --no-resume
```

5. 播放脚本语法验证，必要时手动运行指定 checkpoint。

## 实施顺序

1. 新增 `config.py`。
2. 新增 `env_factory.py` 并替换重复环境创建逻辑。
3. 新增 `replay_buffer.py`。
4. 改造 `Mario` 使用 ReplayBuffer 和配置项。
5. 新增 `trainer.py`，迁移单环境和多环境训练循环。
6. 精简 `main.py`、`main_vector.py`。
7. 更新 `play_trained.py`。
8. 运行语法检查和短训验证。

## 验收标准

- `python -m py_compile *.py` 通过。
- `main_vector.py` 支持 `--num-envs`、`--episodes`、`--no-resume`。
- 多环境训练可以记录真实 episode reward/length，并为 loss/Q 提供合理的 rolling 指标。
- replay buffer 不再以 float32 形式长期保存状态。
- checkpoint 文件名包含 step 和 episode，且 `latest.chkpt` 继续更新。
- 播放和训练使用同一套环境预处理函数。
