# Training Diagnostics and Progress Reward Design

Date: 2026-06-21

## Goal

Improve MadMario training so the agent is encouraged to move right and the training/evaluation output makes policy quality easier to diagnose.

This builds on the existing stuck-penalty work. The new changes should help answer these questions from logs alone:

- Is Mario getting farther to the right?
- Is the best checkpoint actually the one that made the most progress?
- Is the policy collapsing into `NOOP` or one overused action?
- Is the reward signal explicitly encouraging rightward progress?

This iteration will not save evaluation screenshots or videos. The diagnostics will be metric-based only.

## Scope

Included:

- Add `mean_max_x_pos` and a composite `score` to evaluation results.
- Save best checkpoints by evaluation score instead of raw reward.
- Store best evaluation score, reward, max x position, and flag rate in best checkpoints.
- Log episode action distribution summaries: `noop_rate` and `most_used_action`.
- Add a default-enabled `ProgressReward` wrapper that rewards positive `x_pos` deltas.
- Add CLI/config support for disabling or tuning `ProgressReward`.
- Print a training startup hint when `--movement` is not `right_only`.

Excluded:

- Evaluation screenshots or videos.
- Per-step action trace CSV files.
- Q-value preference logging.
- Removing or remapping actions from the action space.
- Network architecture changes such as NoisyNet or C51.
- Changes to replay buffer, PER, or n-step return logic.

## Recommended Approach

Use metric-based diagnostics plus a small reward-shaping wrapper.

The current agent already has several Rainbow-lite components. More model complexity would make debugging harder before the training signal is clear. The recommended change keeps the architecture stable while improving feedback in two ways:

1. evaluation and logging show whether the policy moves farther and which actions it uses;
2. `ProgressReward` gives a small default positive signal for actual rightward movement.

## Evaluation Metrics

Update `mad_mario.training.evaluation.EvalResult` to include:

```python
mean_reward: float
mean_length: float
mean_max_x_pos: float
flag_rate: float
score: float
```

During each evaluation episode:

- initialize `episode_max_x_pos = 0.0`;
- after each `env.step()`, update it from `info.get("max_x_pos", info.get("x_pos", 0.0))`;
- append it to a list at episode end.

The final score is:

```text
score = mean_max_x_pos + 5000 * flag_rate
```

Rationale:

- `flag_rate` dominates when the model starts clearing the level.
- `mean_max_x_pos` distinguishes policies that make partial progress.
- `mean_reward` remains visible but is no longer the primary best-checkpoint signal.

## Best Checkpoint Selection

Update `run_evaluation()` in `mad_mario.training.loops` to maintain `best_eval_score` instead of `best_eval_reward`.

Current behavior:

```python
if result.mean_reward > best_eval_reward:
    checkpoint_manager.save_best(agent, result.mean_reward, episode=agent.current_episode)
```

New behavior:

```python
if result.score > best_eval_score:
    checkpoint_manager.save_best(agent, result, episode=agent.current_episode)
```

Update `CheckpointManager.save_best()` in `mad_mario.agent.checkpoint` to accept an `EvalResult`-like object or explicit score fields, and store these keys in the checkpoint:

```python
best_eval_score
best_eval_reward
best_eval_max_x_pos
best_eval_flag_rate
```

Terminal output should include:

```text
评估分数=...
平均奖励=...
平均最大 x=...
通关率=...
```

This makes `best.chkpt` reflect actual task progress rather than reward alone.

## Action Distribution Logging

Add episode-level action summaries to `mad_mario.logging.metrics.MetricLogger`.

### Metrics

Record these per episode:

```text
noop_rate
most_used_action
```

Then write rolling means to CSV:

```text
mean_noop_rate
mean_most_used_action
```

The rolling mean uses the same recent-episode window as reward and length.

`mean_most_used_action` is the rounded mean of each episode's dominant action id. It is a lightweight diagnostic, not a categorical histogram.

### Single Environment Training

In `train_single_env_loop()`:

- create `action_counts = np.zeros(agent.action_dim, dtype=np.int32)` at episode start;
- after `action = agent.act(state)`, increment `action_counts[action]`;
- at episode end, compute:

```python
noop_rate = action_counts[0] / max(1, ep_length)
most_used_action = int(np.argmax(action_counts))
```

Pass both values to `logger.log_episode()`.

### Vector Training

In `train_vector_env_loop()`:

- create `ep_action_counts = np.zeros((num_envs, agent.action_dim), dtype=np.int32)`;
- after `actions = agent.act_batch(states)`, increment each environment's selected action count;
- when a sub-environment finishes, compute its `noop_rate` and `most_used_action`, pass them to `logger.log_episode_metrics()`, and reset that row.

### NOOP Assumption

This design assumes action id `0` maps to `NOOP`, which is true for the standard `gym_super_mario_bros.actions.RIGHT_ONLY` ordering and common JoypadSpace action lists. The metric name should remain `noop_rate` because this project currently defaults to `right_only` and uses JoypadSpace.

## ProgressReward Wrapper

Add `ProgressReward` to `mad_mario.env.wrappers`.

### Behavior

The wrapper reads `x_pos` from `info` and adds reward only for positive horizontal progress:

```python
delta_x = x_pos - last_x_pos
reward += progress_reward_scale * max(0.0, delta_x)
```

It should:

- initialize `last_x_pos` on `reset()` when available;
- tolerate missing `x_pos` without changing reward;
- update `last_x_pos` after each valid `x_pos` observation;
- not reward negative movement or no movement.

### Defaults

Add fields to `EnvConfig`:

```python
progress_reward_enabled: bool = True
progress_reward_scale: float = 0.01
```

Add train CLI options:

```text
--no-progress-reward
--progress-reward-scale
```

The user chose to enable `ProgressReward` by default.

### Wrapper Order

Use this order in `mad_mario.env.factory.make_mario_env()`:

```text
JoypadSpace
SkipFrame
ClipReward
ProgressReward
StuckPenalty
GrayScaleObservation
ResizeObservation
NormalizeObservation / FrameStack
```

Rationale:

- `ClipReward` clips only the original environment reward.
- `ProgressReward` and `StuckPenalty` are extra training signals and should not be clipped away.
- Both shaping wrappers need access to raw `info` before observation preprocessing.

## right_only Training Hint

Add a startup hint in the training path, likely near the start of `mad_mario.training.trainer.train()`.

If:

```python
config.env.movement != "right_only"
```

print:

```text
建议先用 --movement right_only 学会稳定向右，再尝试 simple/complex。
```

This is informational only. It should not change configuration or block training.

## CSV Compatibility

The metrics CSV loader should tolerate older CSV files that lack the new columns:

- `mean_max_x_pos`
- `mean_noop_rate`
- `mean_most_used_action`

Missing values should default to `0.0` while loading history.

The existing `mean_max_x_pos` from the stuck-penalty work should be preserved and extended, not duplicated.

## Testing and Verification

Use lightweight checks focused on behavior and compatibility:

1. Run syntax checks:

```bash
python -m compileall src/mad_mario
```

2. Check CLI help:

```bash
python -m mad_mario.cli train --help
```

The help should include:

```text
--no-progress-reward
--progress-reward-scale
```

3. Test `ProgressReward` with a fake env:

- increasing `x_pos` adds `progress_reward_scale * delta_x`;
- equal `x_pos` adds nothing;
- decreasing `x_pos` adds nothing;
- missing `x_pos` does not crash and does not alter reward.

4. Test evaluation scoring with a small fake result or fake env path:

- `mean_max_x_pos` is computed;
- `score = mean_max_x_pos + 5000 * flag_rate`.

5. Test `MetricLogger` output:

- CSV header contains `mean_noop_rate` and `mean_most_used_action`;
- rows include expected values;
- loading older CSV rows without these columns still succeeds.

6. If environment dependencies allow, run a very short smoke test:

```bash
python -m mad_mario.cli train --episodes 1 --no-resume --record-every 1
```

## Success Criteria

The implementation is successful when:

- evaluation reports `mean_max_x_pos` and `score`;
- `best.chkpt` is selected by score and stores all best evaluation fields;
- training CSV records action diagnostics;
- terminal output includes NOOP/action diagnostics;
- `ProgressReward` is enabled by default and can be disabled by CLI;
- no screenshot or video files are created by this feature;
- existing training still runs with `--no-progress-reward` for comparison experiments.
