# Stuck Penalty Training Design

Date: 2026-06-21

## Goal

Reduce the chance that the Mario agent learns a "do nothing" policy during training, and make that failure mode visible in the training logs.

The change should:

- Penalize episodes that stop moving for too long.
- End stuck episodes early so replay memory is not filled with repeated no-progress transitions.
- Record episode-level maximum horizontal progress so training runs can be diagnosed from CSV and terminal output.
- Keep reward clipping enabled by default, while recommending `--no-reward-clip` for training runs where progress reward should remain fully visible.

## Scope

Included:

- A new `StuckPenalty` environment wrapper.
- Configuration and CLI options for enabling/disabling stuck handling and tuning its threshold and penalty.
- Integration into the existing environment wrapper chain.
- Episode-level `max_x_pos` logging for single-environment and vector training.
- CSV and terminal output updates for `mean_max_x_pos`.
- Lightweight tests or checks around wrapper behavior, CLI parsing, and metric output.

Excluded:

- Removing `NOOP` from the action space.
- Adding movement bonuses or idle penalties every step.
- Adding new plots for `max_x_pos`.
- Changing the DQN architecture, replay algorithm, or checkpoint format.
- Persisting extra per-step trajectory data.

## Recommended Approach

Use a small wrapper plus episode-level logging.

This is preferred over reward shaping because the immediate problem is replay pollution from long no-progress episodes, not a full redesign of Mario's reward function. Ending stuck episodes creates a clear negative signal and makes training spend more time on useful attempts. Logging `max_x_pos` gives a simple way to see whether the policy is improving, without adding large trace files or a plotting rewrite.

## Environment Wrapper

Add `StuckPenalty` to `mad_mario.env.wrappers`.

### Responsibilities

The wrapper tracks movement using both `x_pos` and `y_pos` from the environment `info` dictionary.

On `reset()`, it initializes:

- `last_x_pos`
- `last_y_pos`
- `max_x_pos`
- `stuck_steps`

On each `step()`:

1. Call the wrapped environment.
2. Read `x_pos` and `y_pos` from `info`.
3. Update `max_x_pos` when `x_pos` is available.
4. Treat the agent as moving if either position changed by more than `movement_epsilon`.
5. Reset `stuck_steps` to zero when movement occurs.
6. Increment `stuck_steps` when no movement occurs.
7. When `stuck_steps >= max_stuck_steps`:
   - subtract `penalty` from the reward;
   - set `truncated = True`;
   - set `info["stuck"] = True`.
8. Always write diagnostic fields back into `info`:
   - `info["max_x_pos"]`
   - `info["stuck_steps"]`

If either `x_pos` or `y_pos` is unavailable, the wrapper should not trigger stuck termination for that step. It should still preserve known diagnostics, including `max_x_pos` when `x_pos` is available.

### Defaults

Use these defaults:

```python
stuck_max_steps = 120
stuck_penalty = 5.0
movement_epsilon = 0.0
```

With the existing `frame_skip=4`, `120` wrapped environment steps correspond to roughly 480 emulator frames. This is intentionally lenient so short tactical pauses do not end an episode.

## Configuration and CLI

Add fields to `EnvConfig` in `mad_mario.config`:

```python
stuck_penalty_enabled: bool = True
stuck_max_steps: int = 120
stuck_penalty: float = 5.0
```

Add training CLI options:

```text
--no-stuck-penalty
--stuck-max-steps
--stuck-penalty
```

`StuckPenalty` is enabled by default. The disable flag exists for comparison experiments.

The `config_from_train_args()` path should pass these values into `EnvConfig`. The play configuration can use the same defaults unless later evidence shows that stuck truncation interferes with manual playback; playback already runs with the same environment construction path and benefits from consistent diagnostics.

## Wrapper Order

Integrate `StuckPenalty` in `mad_mario.env.factory` after `ClipReward` and before observation preprocessing:

```python
env = JoypadSpace(env, movement_actions)
env = SkipFrame(env, skip=config.frame_skip)
if config.clip_rewards:
    env = ClipReward(env, clip_value=config.reward_clip_value)
if config.stuck_penalty_enabled:
    env = StuckPenalty(
        env,
        max_stuck_steps=config.stuck_max_steps,
        penalty=config.stuck_penalty,
    )
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=config.resize_shape)
```

Placing `StuckPenalty` after `ClipReward` preserves normal reward clipping while preventing the stuck penalty from being clipped back to `-1`.

For training runs, the recommended command remains:

```bash
python -m mad_mario train --no-reward-clip
```

If the policy still stalls, tune the stuck handling:

```bash
python -m mad_mario train \
  --no-reward-clip \
  --stuck-max-steps 90 \
  --stuck-penalty 8.0
```

## Metric Logging

Record `max_x_pos` as an episode-level metric.

### CSV Output

Add a `mean_max_x_pos` column to the metrics CSV header:

```text
episode,step,epsilon,mean_reward,mean_length,mean_loss,mean_q,mean_max_x_pos,time_delta
```

The value is the rolling mean over the same recent episode window already used for reward, length, loss, and Q values.

### Logger State

Add an episode list:

```python
self.ep_max_x_positions = []
```

Extend episode logging methods so callers can pass `max_x_pos`. If no value is provided, default to `0.0` to keep the logger robust.

### Terminal Output

Extend `MetricLogger.record()` output with:

```text
平均最大 x=...
```

No new plot is added in this iteration. CSV and terminal output are enough for diagnosing whether Mario is still stuck near the spawn point.

## Training Loop Integration

### Single Environment

In `train_single_env_loop()`:

- Initialize `ep_max_x_pos = 0.0` at episode start.
- After every environment step, update it from `info`:

```python
ep_max_x_pos = max(ep_max_x_pos, info.get("max_x_pos", info.get("x_pos", 0)))
```

- At episode end, call:

```python
logger.log_episode(max_x_pos=ep_max_x_pos)
```

### Vector Environment

In `train_vector_env_loop()`:

- Maintain per-environment progress:

```python
ep_max_x_positions = np.zeros(config.training.num_envs, dtype=np.float32)
```

- Each vector step reads `max_x_pos` from `infos` when present, falling back to `x_pos` when needed.
- Update each sub-environment's `ep_max_x_positions[env_index]` independently.
- When a sub-environment finishes, pass that value to `logger.log_episode_metrics()` and reset the slot to `0.0`.

This keeps vector accounting aligned with the existing per-environment reward and length arrays.

## Error Handling and Edge Cases

- Missing `x_pos` or `y_pos` should not crash training.
- Missing position info should not trigger false stuck termination.
- `info` should be copied before mutation if necessary to avoid surprising wrappers or vector environments that reuse dictionaries.
- Existing metrics CSV files may not contain the new column. History loading should tolerate older CSV files by defaulting missing `mean_max_x_pos` values to `0.0`.
- If an episode is already `terminated` or `truncated`, `StuckPenalty` should not undo that state; it only adds truncation when the stuck threshold is reached.

## Testing and Verification

Use lightweight verification focused on the new behavior:

1. Create a fake environment for `StuckPenalty` where `x_pos` and `y_pos` remain constant. Verify that reaching the threshold subtracts the penalty, sets `truncated=True`, and writes `info["stuck"]`.
2. Verify that changing either `x_pos` or `y_pos` resets `stuck_steps`.
3. Verify that `max_x_pos` tracks the largest observed horizontal position.
4. Verify that CLI parsing accepts `--no-stuck-penalty`, `--stuck-max-steps`, and `--stuck-penalty`.
5. Verify that a new metrics CSV header contains `mean_max_x_pos`.
6. Run the existing import or test checks available in the project.

## Success Criteria

The implementation is successful when:

- A stationary Mario episode receives a stuck penalty and is truncated after the configured threshold.
- Training can be run with default reward clipping or with `--no-reward-clip`.
- Metrics CSV files include `mean_max_x_pos`.
- Terminal training records include average maximum x position.
- Single-environment and vector training both attribute `max_x_pos` to the correct episode.
- Existing training behavior is unchanged when `--no-stuck-penalty` is used, except for the added logging defaults.
