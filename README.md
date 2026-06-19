
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

| Option | Command | Description |
|--------|---------|-------------|
| **CPU only** | `uv sync --extra cpu` | For CPU-only training |
| **CUDA 12.6** | `uv sync --extra cu126` | For NVIDIA GPUs with CUDA 12.6 |
| **CUDA 13.0** | `uv sync --extra cu130` | For NVIDIA GPUs with CUDA 13.0 |
| **CUDA 13.2** | `uv sync --extra cu132` | For NVIDIA GPUs with CUDA 13.2 |

### 3. Activate Environment

```bash
# Activate the virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

## Running

### Train Mario

To start the **training** process for Mario:

```bash
python main.py
```

This starts the *Double Q-learning* algorithm and logs key training metrics to `checkpoints/`. A copy of `MarioNet` and current exploration rate will be saved automatically.

- **GPU** will be used automatically if available
- Training time: ~20 hours on GPU, ~80 hours on CPU

### Evaluate a Trained Mario

To **evaluate** a trained Mario:

```bash
python replay.py
```

This visualizes Mario playing the game in a window. Performance metrics will be logged to a new folder under `checkpoints/`.

To evaluate a specific checkpoint, modify the `load_dir` path in `Mario.load()` (e.g., `checkpoints/2020-06-06T22-00-00`).

## Project Structure

| File | Description |
|------|-------------|
| **main.py** | Main loop between Environment and Mario agent |
| **agent.py** | Defines agent behavior: experience collection, action selection, policy updates |
| **wrappers.py** | Environment pre-processing (observation resizing, RGB to grayscale, etc.) |
| **neural.py** | Q-value estimator using convolutional neural networks |
| **metrics.py** | `MetricLogger` for tracking training/evaluation performance |
| **tutorial.ipynb** | Interactive tutorial with detailed explanations |

## Key Metrics

During training, the following metrics are tracked (moving average over past 100 episodes):

- **Episode**: Current episode number
- **Step**: Total number of steps Mario has played
- **Epsilon**: Current exploration rate (ε-greedy policy)
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
