# MadMario

PyTorch implementation based on the [official tutorial](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) to build an AI-powered Mario using Deep Reinforcement Learning.

## Set Up

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### 1. Install uv

```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

Choose one of the following options based on your hardware:

| Option        | Command                 | Description                    |
| ------------- | ----------------------- | ------------------------------ |
| **CPU only**  | `uv sync --extra cpu`   | For CPU-only training          |
| **CUDA 12.6** | `uv sync --extra cu126` | For NVIDIA GPUs with CUDA 12.6 |
| **CUDA 13.0** | `uv sync --extra cu130` | For NVIDIA GPUs with CUDA 13.0 |
| **CUDA 13.2** | `uv sync --extra cu132` | For NVIDIA GPUs with CUDA 13.2 |

### 3. Activate Environment

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

## CLI Reference

The CLI provides two subcommands: `train` and `play`.

### Usage

```bash
# If installed as package
mad-mario train [options]
mad-mario play [options]

# Or via module
python -m mad_mario.cli train [options]
python -m mad_mario.cli play [options]
```

### `train` Options

`--config PATH` is a global option (it goes before the subcommand) pointing at a JSON config file; flags given explicitly on the command line override the matching values in that file.

| Argument | Default | Description |
| -------- | ------- | ----------- |
| **Game & Level** | | |
| `--game` | `SuperMarioBros` | Gym environment game name |
| `--world` | `1` | World / level pack number |
| `--level` | `1` | Level number |
| `--version` | `0` | Environment version suffix |
| `--levels` | — | Multi-level training, comma-separated (e.g. `1-1,1-2,1-3`) |
| `--worlds` | — | Multi-world training, supports ranges (e.g. `1-3,5,7-8`); each world automatically uses all levels 1–4 |
| `--movement` | `right_only` | Action space preset — choices: `right_only`, `simple`, `complex` |
| **Training** | | |
| `--timesteps`, `--total-timesteps` | `10000000` | Total training steps (two spellings of the same option) |
| `--learning-rate` | `6.25e-05` | Learning rate |
| `--batch-size` | `64` | Batch size per learning step |
| `--gamma` | `0.99` | Discount factor (γ) |
| `--buffer-size` | `100000` | Experience replay capacity — a **total** transition count, divided by `--num-envs` internally. Costs roughly 53 MiB per 1000 transitions, so check your free RAM before raising it |
| `--learning-starts` | `50000` | Steps of experience to collect before training starts |
| `--train-freq` | `4` | Train every N environment steps |
| `--target-update-interval` | `8000` | Target network update interval |
| `--exploration-fraction` | `0.1` | Fraction of total steps over which ε decays linearly; ignored when `--exploration-steps` is set |
| `--exploration-steps` | — | Absolute ε decay budget: the global step at which ε reaches `--exploration-final-eps`. Unlike the fraction it does not grow with `--timesteps`, and it is re-applied on resume |
| `--exploration-initial-eps` | `1.0` | Initial ε |
| `--exploration-final-eps` | `0.02` | Final ε |
| `--max-grad-norm` | `10.0` | Gradient clipping max norm |
| `--seed` | `42` | Random seed |
| `--n-steps` | `1` | n-step return horizon; values > 1 switch SB3 to `NStepReplayBuffer` |
| **Checkpoint & Logging** | | |
| `--save-root` | `checkpoints` | Checkpoint root directory |
| `--flat-save-root` | *(flag)* | Use save-root directly without the compatibility subdirectory |
| `--run-name` | — | Name for this training run |
| `--checkpoint` | — | Path to a specific checkpoint to resume from |
| `--save-replay-buffer` | *(flag)* | Also save the replay buffer (the file can be very large) |
| `--save-freq` | `100000` | Save a periodic checkpoint every N environment steps |
| `--num-envs` | `cpus // 2` | Number of parallel environments |
| `--eval-freq` | `10000` | Run a greedy evaluation every N environment steps. Each evaluation runs `eval-levels × --eval-episodes` episodes |
| `--eval-episodes` | `3` | Episodes per evaluation level |
| `--eval-levels` | `3` | How many training levels the evaluation covers, picked evenly from the training set. Set to `1` for the old single-level behaviour |
| `--no-play-best` | *(flag)* | Skip the automatic recording whenever a new best model is saved (on by default) |
| `--log-interval` | `20` | Log metrics and refresh charts every N episodes |
| `--log-level` | `info` | Log level — choices: `debug`, `info`, `warning`, `error` |
| **Reward Shaping** | | |
| `--reward-clip` / `--no-reward-clip` | *(off)* | Clip rewards to ±`--reward-clip-value`. Off by default: clipping flattens the −25 death penalty to −1 and the ~+7 per-step progress reward to +1, discarding most of the native signal |
| `--reward-clip-value` | `1.0` | Reward clipping absolute value |
| `--no-stuck-penalty` | *(flag)* | Disable the stuck penalty and early truncation. Stuck rescue stays on; pass `--no-stuck-rescue` as well to disable stuck handling entirely |
| `--stuck-max-steps` | `120` | Consecutive steps without net progress before considered stuck |
| `--stuck-penalty` | `5.0` | Reward deducted when stuck |
| `--no-stuck-rescue` | *(flag)* | Disable the random-action rescue after getting stuck (falls back to ending the episode on the threshold) |
| `--stuck-rescue-max-steps` | `120` | Step budget for the random-action rescue |
| `--stuck-progress-epsilon` | `8.0` | Net x-progress threshold (pixels) within the window; below this counts as no progress |
| `--no-progress-reward` | *(flag)* | Disable the right-movement reward |
| `--progress-reward-scale` | `0.01` | Reward per pixel moved right |
| `--no-score-reward` | *(flag)* | Disable the score-change reward |
| `--score-reward-scale` | `0.01` | Reward per 1 point of game score |
| `--no-flag-reward` | *(flag)* | Disable the flagpole reward |
| `--flag-reward-value` | `50.0` | Flagpole completion reward value |

### Resuming: CLI flags win

`DQN.load` resolves hyperparameters as *caller kwargs > checkpoint data > class
defaults*. The trainer used to pass none of them, so on a resumed run every CLI
hyperparameter — `--learning-rate`, `--buffer-size`, `--batch-size`, `--gamma`,
`--n-steps`, the ε schedule — was silently ignored; only `--timesteps` took
effect, because it is a `learn()` argument rather than a constructor argument.
The trainer now passes the CLI values explicitly, and logs the effective ones.

Structural parameters (`policy`, `policy_kwargs`) are deliberately *not*
overridden: a changed network invalidates the saved weights, so start a fresh run
instead of resuming.

`--exploration-fraction` is relative to the *total* budget, which grows with
`--timesteps`: `0.1` on a 1e9-step run means a 1e8-step anneal, leaving ε ≈ 0.2
after 87M steps. Use `--exploration-steps N` to pin the anneal to an absolute
global step instead.

### `play` Options

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--checkpoint` | *(auto)* | Checkpoint to load and play. Auto-discovery looks in `<save-root>/<compat>/` for the newest periodic checkpoint, falling back to `mario_dqn_final.zip`; playback errors out if neither exists |
| `--sample [N]` | *(off)* | Draw `N` exploration rates (uniform on `[0, 2×ε]`, mean = the checkpoint's ε) and play every level once per draw with ε-greedily sampled actions instead of the argmax. Each draw's videos go in a directory named after its rate (`0-137/1-1.mp4`). Bare `--sample` means `N=1` |
| `--game` | `SuperMarioBros` | Gym environment game name |
| `--world` | `1` | World / level pack number |
| `--level` | `1` | Level number |
| `--version` | `0` | Environment version suffix |
| `--movement` | `right_only` | Action space preset — choices: `right_only`, `simple`, `complex` |
| `--render-mode` | `rgb_array` | Environment render mode — playback always records, so it is forced to `rgb_array` |
| `--save-root` | `checkpoints` | Output root directory |
| `--flat-save-root` | *(flag)* | Use save-root directly without subdirectory |
| `--video-dir` | `<save-root>/videos` | Parent directory for recorded videos; a timestamped subdirectory is created inside it |
| `--video-fps` | `15` | Video frame rate — 15 matches real speed at `frame_skip=4` |
| `--no-show` | *(flag)* | Record only, without the live window. Playthroughs are spread across threads (`cpu_count / 2`) instead of played one at a time |

### Running

### Train Mario

Single environment training:

```bash
python -m mad_mario.cli train
```

Training always runs on parallel environments (there is no single-env mode); `--num-envs` sets how many, defaulting to half the CPU count:

```bash
python -m mad_mario.cli train --num-envs 8
```

Common options:

```bash
python -m mad_mario.cli train \
  --timesteps 2000000 \
  --num-envs 8 \
  --log-interval 20
```

Multi-world training (each world automatically uses all levels 1–4):

```bash
# Worlds 1 and 3 → 1-1..1-4, 3-1..3-4（共 8 关）
python -m mad_mario.cli train --worlds 1,3

# Range: worlds 1 through 4 → 共 16 关
python -m mad_mario.cli train --worlds 1-4

# Mixed: worlds 1-3, 5, and 7-8 → 共 24 关
python -m mad_mario.cli train --worlds 1-3,5,7-8
```

Training resumes automatically from the newest periodic checkpoint in `<save-root>` (or its `snapshots/` subdirectory), falling back to `mario_dqn_final.zip`. Pass `--checkpoint` to resume from a specific file instead. There is currently no flag to force a from-scratch run — clear the save directory first, or point `--save-root` somewhere new.

If the project is installed, the console script is also available:

```bash
mad-mario train
mad-mario play --checkpoint checkpoints/worlds-1-8-v0_right_only/mario_dqn_final.zip
```

### Play a Trained Mario

```bash
# 单个关卡：边看边录
python -m mad_mario.cli play --checkpoint checkpoints/worlds-1-8-v0_right_only/mario_dqn_final.zip

# 指定关卡与输出目录
python -m mad_mario.cli play --checkpoint checkpoints/worlds-1-8-v0_right_only/mario_dqn_final.zip \
    --levels 1-1,1-2 --video-dir out/videos

# 只录制，不弹窗
python -m mad_mario.cli play --checkpoint checkpoints/worlds-1-8-v0_right_only/mario_dqn_final.zip --no-show
```

Playback shows the game in a window while recording it: each level gets its own `.mp4` named after the level (`1-1.mp4`, `1-2.mp4`), written into a **fresh timestamped directory** so replaying the same model never overwrites the last run:

```text
<save-root>/videos/
  20260912_143022/
    1-1.mp4
    1-2.mp4
  20260912_151047/      # 另一次播放，各存各的
    1-1.mp4
    1-2.mp4
```

The directory is named after local start time (`YYYYMMDD_HHMMSS`); playing twice within the same second appends a counter (`20260912_143022_2`) rather than sharing a directory. `--video-dir` only changes where the parent goes — the timestamped directory is still created inside it.

Every run records **the same playthrough**: playback takes the argmax action, and the emulator always resets Mario to the same starting state, so the same inputs produce the same trajectory frame for frame. Playing twice gives two byte-identical videos — on the levels that end before Mario gets stuck. When he does get stuck, the stuck handler takes over the action and samples it from the action space rather than from the policy, which is the only way out of a loop the policy keeps walking back into; that sample is drawn from a per-environment RNG, so those clips differ run to run. In a 32-level greedy run, 27 came out byte-identical across two runs and the same 5 did not — and the same 5 across two runs of the serial path, which is how you can tell it is the rescue and not the playback.

`--sample N` is what breaks that tie — it draws `N` exploration rates, uniformly from `[0, 2 × ε]` where `ε` is the rate stored in the checkpoint, and plays **every level once per draw**:

```bash
python -m mad_mario.cli play --checkpoint <model.zip> --levels 1-1,1-2 --sample 3
```

```text
<save-root>/videos/20260912_143022/
  0-052/            # 抽到 0.052，这三遍里最贪心的一遍
    1-1.mp4
    1-2.mp4
  0-198/            # 抽到 0.198 = checkpoint 原值，也就是均值
    1-1.mp4
    1-2.mp4
  0-371/            # 抽到 0.371，最随机的一遍
    1-1.mp4
    1-2.mp4
```

The rate names the **directory**, not the file: the directory says how random that playthrough was, the filename says which level it is, and neither has to carry both. The dot becomes a dash (`0-198`, not `0.198`) because a directory name with a dot in it reads like it has an extension. A directory therefore holds what one random intensity produced across every level, which is what makes the clips inside it worth comparing.

The range is symmetric around the checkpoint's own rate, so its mean is that rate: playing many times shows you the model you actually have, rather than a systematically wilder or tamer one. Each playthrough reassigns the rate on the loaded model before it starts, and playback logs every drawn value alongside the range, so an unusually wild clip is traceable to a number rather than a mystery.

A bare `--sample` means `--sample 1`. Greedy playback has no exploration rate to show, so it stays flat — `videos/20260912_143022/1-1.mp4`, no subdirectory. Two draws only collide if they round to the same three decimals (about a 1% chance at `N=3`); when that happens the second playthrough's clips get `_2` appended (`1-1_2.mp4`) instead of overwriting the first, which is exactly the kind of loss nobody would ever diagnose.

With a window, the runs of one level share a single environment instance — it is reset between them, which is what makes each run a fresh game rather than a continuation. Without one they do not: see below.

`--sample` also reseeds the random number generator from OS entropy after loading: a checkpoint carries the training `seed` (this repository's stores `42`), and `DQN.load` replays it, which would otherwise make separate runs draw the identical "random" rates and sample the identical sequence — producing byte-identical videos again. Each worker thread reseeds once more after loading its own copy, for the same reason between threads: every one of them loading the same archive would otherwise replay one sequence between them.

The sampling strength comes from the checkpoint, which is a training artifact rather than a playback choice: this repository's model stores `0.198`, so each draw comes from `[0, 0.396]`, i.e. somewhere between always taking the argmax and having roughly two steps in five be random. Episode length and reward vary as a result; in one real run three playthroughs of 1-1 ended at 38, 42 and 205 steps. To see even more of them, raise `N` or run the command again — the timestamped directory keeps each batch from overwriting the last.

A checkpoint whose exploration rate has decayed to `0` (or a model that carries no such rate at all) has nothing to draw from: playback logs a warning, and every run comes out greedy and identical, falling back to the flat layout since there is no rate to name a directory after.

The window is paced at `--video-fps`, so what you watch matches the video's playback speed.

It stops when you close the window or press `Ctrl+C`; either way the partial recording is kept.

Video needs pixel frames, so playback always forces `rgb_array` rendering regardless of `--render-mode`. A clip contains one frame per agent step plus the opening frame, so its length is `steps / video_fps` seconds.

Pass `--no-show` to record without a window — that is also what happens automatically on a machine with no display (a server or an SSH session), where playback logs a warning and records the video only. Either way each frame is rendered once and used for both output.

A window has to show one frame after another, so it cannot be sped up. Without one there is nothing to keep in step, and playback spreads the queue across threads instead: `min(cpu_count / 2, playthroughs)` of them, each with its own copy of the model and its own environment. Half the cores are left to whatever else the machine is doing, and it is capped by the amount of work rather than by the machine — two runs do not open twelve threads to stand around. It pays off on the batch runs: 32 greedy levels on a 24-core machine went from 35s to 13s, and a `--sample 2 --levels 1-1,1-2,1-3,1-4` batch recorded its 8 playthroughs in 6.4s.

Each thread loads a private model rather than sharing one. `DQN.predict` reads `model.exploration_rate` on every call and each playthrough rewrites it to the rate it drew, so a shared instance would have threads overwriting each other's number — and the directory names, which are that number, would quietly become fiction. The environment is per playthrough too: which playthrough lands on which thread is up to the scheduler, and caching environments across threads means owning their handoff and cleanup for the sake of about 160 milliseconds.

`Ctrl+C` behaves differently depending on which path you are on. With a window it stops between levels, immediately. Without one, the playthroughs already running finish first — a thread cannot be interrupted from outside — so count on up to `cpu_count / 2` of them still completing; the clips they recorded are kept either way.

### Automatic Recording of the Best Model

While training, every time the evaluation callback saves a new best model, playback runs on it automatically and records **all the levels training uses** — no window, video only. Results land next to the checkpoint, one directory per training stage:

```text
checkpoints/<compat>/
  best_model/
    best_model.zip
  best_model_videos/
    step_000160000/      # 第 16 万步刷新时的录制
      1-1.mp4
      1-2.mp4
    step_000830000/
      ...
```

The `step_XXXX` directory name is the step count when the best score improved, so clips from different stages of training can be compared side by side. These directories are stamped by training step rather than by wall-clock time — the timestamped-directory scheme described above applies to the `play` command only. Recording uses unshaped playback settings (reward shaping and the stuck penalty off, stuck rescue still on) — it shows what the agent has actually learned, not what the training rewards push it toward.

Recording failures only log a warning and never interrupt training: a full disk or a broken encoder does not cost you the run. Pass `--no-play-best` to turn the automatic recording off entirely; evaluation and best-model saving are unaffected.

## Output Files

Everything lands under the run directory `<save-root>/<compatibility-folder>/`, where the folder encodes the level selection and movement preset (e.g. `worlds-1-8-v0_right_only`); `--flat-save-root` writes into `<save-root>` itself:

```text
checkpoints/worlds-1-8-v0_right_only/
  mario_dqn_final.zip          # 收尾保存的最终模型
  snapshots/                   # 周期保存：mario_dqn_<steps>_steps.zip
  best_model/
    best_model.zip             # 评估成绩最好的模型
  best_model_videos/           # 最优模型刷新时的自动录制（见上）
    step_000160000/1-1.mp4
  eval_logs/
    evaluations.npz            # 每次评估的奖励/长度
  tensorboard/
  logs/
    train_20260912_143022.log  # 训练期间的标准输出镜像
  videos/                      # play 命令的输出（按时间戳分目录）
    20260912_151047/1-1.mp4
  latest_metrics.csv
  latest_reward.png
  latest_length.png
  latest_loss.png
  latest_q.png
```

The four `latest_*.png` charts are redrawn every `--log-interval` episodes and read their history back from `latest_metrics.csv`, so a resumed run continues the curves instead of restarting them from zero.

## Project Structure

| Path                                      | Responsibility                                                 |
| ----------------------------------------- | -------------------------------------------------------------- |
| `src/mad_mario/cli.py`                    | Command-line entry point; dispatches to `train` and `play`     |
| `src/mad_mario/config.py`                 | Argument parsing, config dataclasses, checkpoint resolution    |
| `src/mad_mario/env/factory.py`            | Mario environment and vector environment creation              |
| `src/mad_mario/env/wrappers.py`           | Environment preprocessing wrappers                             |
| `src/mad_mario/training/trainer.py`       | SB3 DQN assembly, callbacks, and the training loop             |
| `src/mad_mario/training/best_playback.py` | Records a video whenever the evaluation finds a new best model |
| `src/mad_mario/playback/runner.py`        | Episode playback loop and per-level video wiring               |
| `src/mad_mario/playback/video.py`         | Frame-to-mp4 recorder                                          |
| `src/mad_mario/playback/viewer.py`        | Live matplotlib window used while recording                    |
| `src/mad_mario/logging/metrics.py`        | CSV metrics and plot generation                                |
| `src/mad_mario/logging/setup.py`          | Logging setup and stdout mirroring                             |

Source code uses the `src/` layout. New code should import from `mad_mario` directly. The old root-level training/import shims have been removed.

The Q-network itself is not in this repository: it is SB3's `CnnPolicy`, i.e. the NatureCNN feature extractor from the Nature 2015 DQN paper, followed by a 512-unit head and a linear Q-value output. See [Key Metrics](#key-metrics) and the `train` options above; `policy_kwargs` (currently unset) is the hook for swapping in a different architecture.

## Key Metrics

During training, the following metrics are tracked as moving averages over recent episodes:

- **Episode**: Current episode number
- **Step**: Total number of environment steps Mario has played
- **Epsilon**: Current exploration rate for the ε-greedy policy
- **MeanReward**: Average reward per episode
- **MeanLength**: Average episode length
- **MeanLoss**: Average training loss
- **MeanQValue**: Average predicted Q-value

## Pre-trained Model

Checkpoint for a trained Mario agent:
[Google Drive Download](https://drive.google.com/file/d/1RRwhSMUrpBBRyAsfHLPGt1rlYFoiuus2/view?usp=sharing)

## Resources

- **Deep Reinforcement Learning with Double Q-learning** - Hado V. Hasselt et al, NIPS 2015: [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)
- **OpenAI Spinning Up tutorial**: [spinningup.openai.com](https://spinningup.openai.com/en/latest/)
- **Reinforcement Learning: An Introduction** - Richard S. Sutton et al.: [Online Book](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- **super-mario-reinforcement-learning** - GitHub: [sebastianheinz/super-mario-reinforcement-learning](https://github.com/sebastianheinz/super-mario-reinforcement-learning)
- **Deep Reinforcement Learning Doesn't Work Yet**: [alexirpan.com](https://www.alexirpan.com/2018/02/14/rl-hard.html)
