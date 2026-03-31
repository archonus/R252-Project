# Quantum Grokking Experiments

This repository contains experiments around grokking behavior in quantum and hybrid models, with a focus on:

- Quantum models on modular arithmetic (Cayley table) tasks
- Quantum models on parity tasks
- Low-depth quantum state/classification experiments on MNIST-like data
- Quantum transformer experiments

## Repository Layout

- `quantum_grokking.py`: Main training entry point for Cayley-table quantum grokking with W&B sweeps.
- `resume_training.py`: Resume a previous W&B run from checkpoint and continue training with updated hyperparameters.
- `parity_grokking.py`: Grokking experiments on parity datasets.
- `low_depth_mnist.py`: Low-depth quantum circuit MNIST pipeline.
- `plot_sweep.py`: Utilities to visualize W&B sweep outputs.
- `plot_distributions.py`: Plot/inspect parameter or training distributions.
- `cayley_table.py`, `parity_dataset.py`: Dataset generation/helpers.
- `quantum_transformer/`: Quantum transformer experiments, forked from [QuantumTransformers](https://github.com/salcc/QuantumTransformers), for MNIST and Cayley table experiments. Kept as a project since requirements clash with main project, so separate virtual environment required. 

## Requirements

- Python 3.10+
- Recommended: `uv` for dependency management
- Optional but expected for most training runs: Weights & Biases account

Core dependencies are declared in `pyproject.toml`.

## Setup

### Option 1: uv (recommended)

```bash
uv sync
```

Run scripts with:

```bash
uv run python quantum_grokking.py --help
```

### Option 2: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Environment Variables

Most training scripts log to W&B and expect a `.env` file in the repository root:

```env
WANDB_ENTITY=your_wandb_entity
WANDB_PROJECT=your_wandb_project
```

You may also need to authenticate W&B once:

```bash
wandb login
```

## Usage Examples

### 1) Quantum grokking on modular arithmetic

```bash
uv run python quantum_grokking.py \
  --epochs 200 \
  --prime 13 \
  --depth 8 \
  --train_frac 0.8 \
  --wd_coef 1e-4 1e-3 1e-2
```

Angle encoding variant:

```bash
uv run python quantum_grokking.py --angle_encoding True
```

### 2) Resume training from a previous W&B run

```bash
uv run python resume_training.py \
  --run_id <wandb_run_id> \
  --epochs 100 \
  --wd_coef 1e-4 1e-3
```

