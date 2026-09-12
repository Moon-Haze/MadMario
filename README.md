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
| **Environment** | | |
| `--vector` | *(flag)* | Enable vectorized (parallel) environments |
| `--num-envs` | `cpus // 2` | Number of parallel environments |
| `--no-reward-clip` | *(flag)* | Disable reward clipping |
| `--reward-clip-value` | `1.0` | Reward clipping absolute value |
| `--no-stuck-penalty` | *(flag)* | Disable stuck penalty and early truncation |
| `--stuck-max-steps` | `120` | Consecutive steps without movement before considered stuck |
| `--stuck-penalty` | `5.0` | Extra reward deduction when stuck |
| `--no-progress-reward` | *(flag)* | Disable right-movement reward |
| `--progress-reward-scale` | `0.01` | Reward coefficient per pixel moved right |
| `--normalize-observation` | *(flag)* | Normalize observations (default keeps uint8) |
| **Training Loop** | | |
| `--episodes` | `20000` | Total training episodes |
| `--batch-size` | `32` | Batch size per learning step |
| `--burnin` | `50000` | Experiences to collect before training starts |
| `--learning-rate` | `0.0000625` | Initial learning rate |
| `--lr-warmup-steps` | `10000` | Learning rate warmup steps |
| `--lr-min-ratio` | `0.1` | Minimum LR as ratio of initial LR |
| `--gamma` | `0.99` | Discount factor (γ) |
| `--exploration-decay-steps` | `2000000` | ε-greedy epsilon linear decay steps |
| `--exploration-min` | `0.02` | Minimum exploration rate |
| `--gradient-clip-norm` | `10.0` | Gradient clipping max norm |
| `--n-step` | `3` | N-step return (number of bootstrap steps) |
| **Network & Algorithm** | | |
| `--replay-capacity` | `100000` | Experience replay buffer capacity |
| `--noisy-std-init` | `0.5` | NoisyNet noise standard deviation initial value (set to `0` for plain DuelingDQN + ε-greedy) |
| `--no-noisy` | *(flag)* | Disable NoisyNet, fall back to DuelingDQN + ε-greedy |
| `--per-alpha` | `0.6` | Prioritized Experience Replay (PER) priority exponent |
| `--per-beta-start` | `0.4` | PER importance-sampling β initial value |
| `--per-beta-frames` | `2000000` | PER β annealing frames |
| `--per-epsilon` | `1e-6` | PER priority epsilon |
| **Checkpoint & Logging** | | |
| `--save-root` | `checkpoints` | Checkpoint root directory |
| `--flat-save-root` | *(flag)* | Use save-root directly without version/movement subdirectory |
| `--checkpoint` | — | Path to a specific checkpoint to resume from |
| `--no-resume` | *(flag)* | Ignore latest checkpoint, start training from scratch |
| `--keep-runs` | *(flag)* | Keep a timestamped per-run copy of artifacts |
| `--run-name` | — | Custom run directory name (used with `--keep-runs`) |
| `--record-every` | `20` | Log metrics, refresh charts, and save every N episodes |
| `--eval-every` | `50` | Run greedy evaluation every N episodes |
| `--eval-episodes` | `3` | Number of episodes per evaluation |
| `--no-play-best` | *(flag)* | Skip the automatic recording whenever a new best model is saved (on by default) |
| `--save-every` | `500000` | Save checkpoint every N environment steps |

### `play` Options

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--checkpoint` | `checkpoints/<compat>/latest.chkpt` | Checkpoint to load and play |
| `--episodes` | `1` | Number of episodes to play |
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
| `--no-show` | *(flag)* | Record only, without the live window |

### Running

### Train Mario

Single environment training:

```bash
python -m mad_mario.cli train
```

Vectorized training:

```bash
python -m mad_mario.cli train --vector
```

Common options:

```bash
python -m mad_mario.cli train \
  --episodes 40000 \
  --vector \
  --num-envs 8 \
  --record-every 20 \
  --save-every 500000
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

By default, training resumes from `checkpoints/latest.chkpt` if it exists. To start from scratch:

```bash
python -m mad_mario.cli train --no-resume
```

To keep a timestamped copy of the current run in addition to the latest files:

```bash
python -m mad_mario.cli train --keep-runs
```

If the project is installed, the console script is also available:

```bash
mad-mario train
mad-mario train --vector
```

### Play a Trained Mario

```bash
# 单个关卡：边看边录
python -m mad_mario.cli play --checkpoint checkpoints/latest.chkpt

# 指定关卡与输出目录
python -m mad_mario.cli play --checkpoint checkpoints/latest.chkpt \
    --levels 1-1,1-2 --video-dir out/videos

# 只录制，不弹窗
python -m mad_mario.cli play --checkpoint checkpoints/latest.chkpt --no-show
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

The window is paced at `--video-fps`, so what you watch matches the video's playback speed.

It stops when you close the window or press `Ctrl+C`; either way the partial recording is kept.

Video needs pixel frames, so playback always forces `rgb_array` rendering regardless of `--render-mode`. A clip contains one frame per agent step plus the opening frame, so its length is `steps / video_fps` seconds.

Pass `--no-show` to record without a window — that is also what happens automatically on a machine with no display (a server or an SSH session), where playback logs a warning and records the video only. Either way each frame is rendered once and used for both output.

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

Default output keeps only the latest training artifacts:

```text
checkpoints/
  latest.chkpt
  latest_metrics.csv
  latest_reward.png
  latest_length.png
  latest_loss.png
  latest_q.png
```

With `--keep-runs`, the trainer also writes a per-run copy:

```text
checkpoints/runs/<timestamp>/
  checkpoint.chkpt
  metrics.csv
  reward.png
  length.png
  loss.png
  q.png
```

The trainer no longer creates many `mario_net_step_*.chkpt` backup files. Each target location keeps one current checkpoint.

## Project Structure

| Path                                        | Description                                                     |
| ------------------------------------------- | --------------------------------------------------------------- |
| `src/mad_mario/cli.py`                      | Entry point: parse args, set up logging, dispatch to train/play |
| `src/mad_mario/config.py`                   | Dataclass configs and CLI argument parsing                      |
| `src/mad_mario/playback/runner.py`          | Playback: run a checkpoint over levels and record it            |
| `src/mad_mario/playback/video.py`           | Frame-to-mp4 recorder                                           |
| `src/mad_mario/playback/viewer.py`          | Live playback window and GUI-availability check                 |
| `src/mad_mario/training/best_playback.py`   | Records the best model whenever evaluation improves it          |
| `src/mad_mario/agent/mario.py`              | Agent behavior: action selection, replay caching, DQN updates   |
| `src/mad_mario/agent/checkpoint.py`         | Checkpoint loading and saving                                   |
| `src/mad_mario/agent/replay_buffer.py`      | Experience replay buffer                                        |
| `src/mad_mario/env/factory.py`              | Mario environment and vector environment creation               |
| `src/mad_mario/env/wrappers.py`             | Environment preprocessing wrappers                              |
| `src/mad_mario/models/mario_net.py`         | Q-value CNN model                                               |
| `src/mad_mario/training/trainer.py`         | Training component assembly                                     |
| `src/mad_mario/training/loops.py`           | Single-env and vector-env training loops                        |
| `src/mad_mario/training/artifacts.py`       | latest/runs output path management                              |
| `src/mad_mario/logging/metrics.py`          | CSV metrics and plot generation                                 |

Source code uses the `src/` layout. New code should import from `mad_mario` directly. The old root-level training/import shims have been removed.

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
