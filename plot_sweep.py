"""Plot metrics from a wandb sweep, colored by hyperparameter config."""

import os

import wandb
import matplotlib.pyplot as plt
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────
# CONFIGURE THESE
# ──────────────────────────────────────────────────────────────────────

ENTITY = os.getenv("WANDB_ENTITY")  # e.g. "my_username_or_team"
PROJECT = os.getenv("WANDB_PROJECT")  # e.g. "my_project_name"

# Specify ONE of these. Sweep ID takes priority if both are set.
SWEEP_ID = "3tiwiylz"  # e.g. "abc123xy" 
GROUP = ""                 # e.g. "my_experiment_group"

# Metrics to plot (each gets a different line style)
METRICS = ["train_accuracy", "val_accuracy"]
METRIC_NAMES = ["Training Accuracy", "Validation Accuracy"]  # for legend (optional, defaults to raw metric keys)

# Hyperparameters to show in the legend (each unique combo gets a color)
LEGEND_HPARAMS = ["wd_coef", "train_frac"]  # e.g. ["learning_rate", "batch_size"]
LEGEND_NAMES = ["$\\lambda$", "$\\alpha$"]
# X-axis key (usually "epoch" or "_step")
X_KEY = "epoch"

# Running average window size (set to 1 to disable smoothing)
SMOOTH_WINDOW = 10

# Max x-axis value (set to None for no limit)
MAX_X = None

# Plot title
TITLE = "Sweep over Weight Decay and Training Fraction"

# Font sizes
AXIS_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 13
TITLE_FONTSIZE = 16

# ──────────────────────────────────────────────────────────────────────

# Line styles cycle: one per metric
LINE_STYLES = ["--", "-", ":", "-."]

# Color palette for different HP configs
COLORS = plt.cm.tab10.colors


def smooth(values, window):
    """Simple running average. Returns list of same length (shorter windows at edges)."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i+1]) / (i - start + 1))
    return out


def fetch_runs():
    """Fetch runs from a sweep or group."""
    api = wandb.Api()
    if SWEEP_ID and SWEEP_ID != "YOUR_SWEEP_ID":
        sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")
        return list(sweep.runs)[:4]  # only return the first 4 runs
    elif GROUP:
        return api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP})
    else:
        raise ValueError("Set either SWEEP_ID or GROUP at the top of this file.")


def config_key(run):
    """Extract the legend-worthy hyperparameters as a hashable tuple."""
    return tuple((hp, run.summary.get(hp, run.config.get(hp))) for hp in LEGEND_HPARAMS)


def config_label(key):
    """Format a config tuple into a readable legend string."""
    parts = []
    for idx, (hp, val) in enumerate(key):
        display_name = LEGEND_NAMES[idx] if idx < len(LEGEND_NAMES) else hp
        if isinstance(val, (int, float)):
            formatted_val = f"{val:.0e}" if val < 0.001 else round(val, 3)
        else:
            formatted_val = val
        parts.append(f"{display_name}={formatted_val}")
    return ", ".join(parts)


def main():
    runs = fetch_runs()

    # Group runs by their HP config
    groups = defaultdict(list)
    for run in runs:
        if run.state == "finished":
            groups[config_key(run)].append(run)

    # Debug: show discovered configs so you can verify LEGEND_HPARAMS are correct
    print(f"Found {sum(len(v) for v in groups.values())} finished runs in {len(groups)} config groups:")
    for k, v in groups.items():
        print(f"  {config_label(k)}  ({len(v)} runs)")
    if len(groups) <= 1:
        # Print everything we can find to help debug
        first_run = next((r for r in runs if r.state == "finished"), None)
        if first_run:
            print(f"\nrun.config: {dict(first_run.config)}")
            print(f"run.summary keys: {list(first_run.summary.keys())}")
            print(f"run.name: {first_run.name}")
            print(f"run.tags: {first_run.tags}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Assign one color per unique HP config
    config_keys = sorted(groups.keys(), key=lambda k: tuple(v for _, v in k))
    color_map = {k: COLORS[i % len(COLORS)] for i, k in enumerate(config_keys)}

    for cfg_key in config_keys:
        color = color_map[cfg_key]
        for run in groups[cfg_key]:
            # scan_history returns a list of dicts; collect into per-key lists
            rows = list(run.scan_history(keys=[X_KEY] + METRICS))
            if not rows:
                continue
            if MAX_X is not None:
                rows = [r for r in rows if X_KEY in r and r[X_KEY] <= MAX_X]
            xs = [r[X_KEY] for r in rows if X_KEY in r]
            for j, metric in enumerate(METRICS):
                ys = [r.get(metric) for r in rows if X_KEY in r]
                if not any(y is not None for y in ys):
                    continue
                ys = smooth(ys, SMOOTH_WINDOW)
                linestyle = LINE_STYLES[j % len(LINE_STYLES)]
                ax.plot(xs, ys, color=color, linestyle=linestyle, alpha=0.8)

    # Build a compact legend: one entry per color (HP config) + one entry per line style (metric)
    from matplotlib.lines import Line2D
    handles = []
    # HP config entries (colored solid lines)
    for cfg_key in config_keys:
        handles.append(Line2D([0], [0], color=color_map[cfg_key], linestyle="-", label=config_label(cfg_key)))
    # Metric entries (black lines with different styles)
    for j, metric in enumerate(METRICS):
        handles.append(Line2D([0], [0], color="black", linestyle=LINE_STYLES[j % len(LINE_STYLES)], label=metric))
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(X_KEY.title(), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(TITLE, fontsize=TITLE_FONTSIZE)
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("sweep_plots", exist_ok=True)
    filename = f"sweep_plots/{GROUP or SWEEP_ID}_{'_'.join(METRICS)}.png"
    fig.savefig(filename, dpi=150)
    print(f"Saved to {filename}")
    plt.show()


if __name__ == "__main__":
    for SWEEP_ID, SMOOTH_WINDOW in zip(["n1cqfd0s","3tiwiylz", "w0cccj57"], [200, 10, 1]):
        main()
