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
| `--worlds` | — | Multi-world training, comma-separated (e.g. `1,3`); combines with `--levels`/`--level` to generate world × level grids |
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
| `--render-mode` | `rgb_array` | Environment render mode (uses matplotlib display) |
| `--save-root` | `checkpoints` | Output root directory |
| `--flat-save-root` | *(flag)* | Use save-root directly without subdirectory |

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

Multi-world training (generates world × level grid):

```bash
# Worlds 1 and 3, level 1 in each (default)
python -m mad_mario.cli train --worlds 1,3

# Worlds 1 and 2, levels 1 and 2 → 1-1, 1-2, 2-1, 2-2
python -m mad_mario.cli train --worlds 1,2 --levels 1,2

# Worlds 1,2,3 and explicit level IDs
python -m mad_mario.cli train --worlds 1,2,3 --levels 1,2
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
python -m mad_mario.cli play --checkpoint checkpoints/latest.chkpt
```

The play command uses `rgb_array` rendering by default and displays frames through matplotlib, which works around common `nes-py` human-window issues on Windows + Python 3.13.

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

| Path                                   | Description                                                   |
| -------------------------------------- | ------------------------------------------------------------- |
| `src/mad_mario/cli.py`                 | Unified CLI for training and playback                         |
| `src/mad_mario/config.py`              | Dataclass configs and CLI argument parsing                    |
| `src/mad_mario/agent/mario.py`         | Agent behavior: action selection, replay caching, DQN updates |
| `src/mad_mario/agent/checkpoint.py`    | Checkpoint loading and saving                                 |
| `src/mad_mario/agent/replay_buffer.py` | Experience replay buffer                                      |
| `src/mad_mario/env/factory.py`         | Mario environment and vector environment creation             |
| `src/mad_mario/env/wrappers.py`        | Environment preprocessing wrappers                            |
| `src/mad_mario/models/mario_net.py`    | Q-value CNN model                                             |
| `src/mad_mario/training/trainer.py`    | Training component assembly                                   |
| `src/mad_mario/training/loops.py`      | Single-env and vector-env training loops                      |
| `src/mad_mario/training/artifacts.py`  | latest/runs output path management                            |
| `src/mad_mario/logging/metrics.py`     | CSV metrics and plot generation                               |

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
