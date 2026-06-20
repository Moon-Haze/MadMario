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

## Running

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
