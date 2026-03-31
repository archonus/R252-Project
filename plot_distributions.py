"""Plot distributions of learned parameters from the latest checkpoint of each run in a sweep."""

import os
import argparse

import wandb
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

ENTITY = os.getenv("WANDB_ENTITY")
PROJECT = os.getenv("WANDB_PROJECT")

parser = argparse.ArgumentParser(description="Plot parameter distributions from a W&B sweep's latest checkpoints.")
parser.add_argument("--sweep_id", type=str, help="W&B sweep ID")
parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins (default: 50)")
parser.add_argument("--legend_hparams", type=str, nargs="+", default=["weight_decay"],
                    help="Hyperparameters to show in legend (default: weight_decay)")
args = parser.parse_args()


def fetch_runs(sweep_id):
    """Fetch finished runs from a sweep."""
    api = wandb.Api()
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")
    return [r for r in sweep.runs if r.state == "finished"]


def load_params(run):
    """Download the latest model artifact from a run and return the params array."""
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if not artifacts:
        return None
    artifact = artifacts[-1]
    artifact_dir = artifact.download()
    data = np.load(os.path.join(artifact_dir, "latest_model.npz"), allow_pickle=True)
    return data["params"]


def config_label(run, hparams):
    """Format a run's config into a readable legend string."""
    parts = []
    for hp in hparams:
        val = run.config.get(hp, run.summary.get(hp))
        if isinstance(val, float):
            val = round(val, 6)
        parts.append(f"{hp}={val}")
    return ", ".join(parts)


def main():
    runs = fetch_runs(args.sweep_id)
    print(f"Found {len(runs)} finished runs in sweep {args.sweep_id}")

    out_dir = os.path.join("sweep_plots", args.sweep_id)
    os.makedirs(out_dir, exist_ok=True)

    for run in runs:
        params = load_params(run)
        if params is None:
            print(f"  Skipping {run.name}: no model artifact found")
            continue

        label = config_label(run, args.legend_hparams)
        mean = float(np.mean(np.abs(params)))
        std = float(np.std(np.abs(params)))
        print(f"  {run.name} ({label}): (abs) mean={mean:.6f}, (abs) std={std:.6f}, n_params={params.size}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(params, bins=args.bins, alpha=0.7)
        ax.set_xlim(-4, 4)
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}\n(abs) mean={mean:.4f}, (abs) std={std:.4f}, n={params.size}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = os.path.join(out_dir, f"{run.name}_param_distribution.png")
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"  Saved to {filename}")


if __name__ == "__main__":
    main()
