# Imports
# get_ipython().run_line_magic('matplotlib', 'inline')
import os
import operator
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from cayley_table import CayleyTableDataset
import jax
from jax import numpy as jnp
import numpy
import optax
import wandb

import argparse
import json

from dotenv import load_dotenv



def is_running_from_ipython():
    from IPython import get_ipython
    return get_ipython() is not None

defaults = {
    "epochs": 100,
    "wd_coef": [1e-4, 1e-3, 1e-2],
    "wd_schedule": None,  # e.g. '{"0":1e-2,"100":1e-3}' — epoch->wd mapping; overrides wd_coef
    "noise_scale": [0.0],
    "dropout_rate": [0.0],
    "sweep_count": None,
    "train_frac": [0.8],
    "test_frac": 0.0,
    "angle_encoding": False,
    "spectral_reg_coef": [0.0],
    "prime": 5,
    "runs_per_config": 1,
    "depth": 8,
    "period_scale": [0.0],
}

if not is_running_from_ipython():
    parser = argparse.ArgumentParser(description="Train quantum grokking model with a W&B sweep.")
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], help=f"Number of training epochs (default: {defaults['epochs']})")
    parser.add_argument("--wd_coef", type=float, default=defaults["wd_coef"], nargs="+", help="Weight decay values for a W&B sweep")
    parser.add_argument("--wd_schedule", type=str, default=defaults["wd_schedule"],
                        help='JSON dict mapping epoch->weight_decay, e.g. \'{"0":0.01,"100":0.001}\'. Overrides --wd_coef when set.')
    parser.add_argument("--noise_scale", type=float, default=defaults["noise_scale"], nargs="+", help="Parameter noise scale values for a W&B sweep")
    parser.add_argument("--dropout_rate", type=float, default=defaults["dropout_rate"], nargs="+", help="Basis-state parameter dropout rates for a W&B sweep")
    parser.add_argument("--sweep_count", type=int, default=defaults["sweep_count"], help="Maximum number of sweep runs for wandb.agent (default: run all)")
    parser.add_argument("--train_frac", type=float, default=defaults["train_frac"], nargs="+", help="Fraction(s) of data to use for training (swept)")
    parser.add_argument("--test_frac", type=float, default=defaults["test_frac"], help="Fraction of data to use for testing (default: 0.1)")
    parser.add_argument("--angle_encoding", type=bool, default=defaults["angle_encoding"], help="Use angle encoding instead of basis state encoding")
    parser.add_argument("--spectral_reg_coef", type=float, default=defaults["spectral_reg_coef"], nargs="+", help="Spectral regularization coefficient (penalizes off-diagonal DFT energy)")
    parser.add_argument("--sweep_name", type=str, default=None, help="Custom name for the W&B sweep (auto-generated if not set)")
    parser.add_argument("--prime", type=int, default=defaults["prime"], help="Prime number p for the Cayley table dataset (default: 5)")
    parser.add_argument("--runs_per_config", type=int, default=defaults["runs_per_config"], help="Number of times to run each hparam config in the sweep (different seeds)")
    parser.add_argument("--depth", type=int, default=defaults["depth"], help="Number of layers in the quantum circuit (default: 8)")
    parser.add_argument("--full_batch_val", action="store_true", help="Use the full validation set as a single batch (no batching)")
    parser.add_argument("--period_scale", type=float, default=defaults["period_scale"], nargs="+", help="Scaling factor for which period branch the parameters are in. E.g. we should only search in [-pi, pi] so if we are outside of that we should be punished.")
    args = parser.parse_args()
else:
    args = argparse.Namespace(**defaults)

PRIME = args.prime
# Generate Cayley table for modular addition mod p

# Set the seed of the dataset throughout all runs for consistency
dataset = CayleyTableDataset(p=PRIME, seed=42, op=operator.add)

print(f"Cayley table (modular addition mod p={PRIME}):")

# Define the hyperparameters
SEED = 42
MODEL_QUBITS = dataset.num_bits * 2
DEPTH = args.depth
N_CLASSES = dataset.p

from types import SimpleNamespace

def make_circuit_config(n_qubits, depth, angle_encoding=False):
    """Compute parameter counts for either circuit variant."""
    n_first_layer = n_qubits
    n_block = 4
    n_layer_blocks = (n_qubits - 1) * depth * n_block
    n_alpha = 2 * (depth + 1) if angle_encoding else 0
    n_network = n_first_layer + n_layer_blocks + n_alpha
    return SimpleNamespace(
        n_qubits=n_qubits,
        depth=depth,
        N_PARAMS_FIRST_LAYER=n_first_layer,
        N_PARAMS_BLOCK=n_block,
        N_PARAMS_LAYER_BLOCKS=n_layer_blocks,
        N_PARAMS_ALPHA=n_alpha,
        N_PARAMS_NETWORK=n_network,
    )

# Switch between circuits here
USE_ANGLE_ENCODING = args.angle_encoding
cfg = make_circuit_config(MODEL_QUBITS, DEPTH, angle_encoding=USE_ANGLE_ENCODING)
N_PARAMS_NETWORK = cfg.N_PARAMS_NETWORK

print("=" * 20)
print(f"Angle encoding: {USE_ANGLE_ENCODING}")
print(f"Total network parameters: {N_PARAMS_NETWORK}")
print("=" * 20)

# Define the model and training functions
dev = qml.device("default.qubit", wires=MODEL_QUBITS)

@jax.jit
@qml.qnode(dev, interface="jax")
def _basis_state_circuit(network_params, x):
    params = iter(network_params)
    qml.BasisState(x[0], wires=range(MODEL_QUBITS // 2))
    qml.BasisState(x[1], wires=range(MODEL_QUBITS // 2, MODEL_QUBITS))

    # First two layers of local RY rotations
    for w in range(MODEL_QUBITS):
        qml.RY(next(params), wires=w)

    # SO(4) building blocks
    for _ in range(DEPTH):
        for j in range(MODEL_QUBITS - 1):
            qml.CNOT(wires=[j, j + 1])
            qml.RY(next(params), wires=j)
            qml.RY(next(params), wires=j + 1)
            qml.CNOT(wires=[j, j + 1])
            qml.RY(next(params), wires=j)
            qml.RY(next(params), wires=j + 1)

    return qml.probs(wires=range(dataset.num_bits))

def basis_state_circuit(network_params, x):
    probs = _basis_state_circuit(network_params, x)[:N_CLASSES]
    return probs / jnp.sum(probs)

# Recover true Z_p coordinates from permuted dataset labels
PI_INV = jnp.argsort(dataset.pi)

# Explicitly compute the number of model parameters
N_PARAMS_FIRST_LAYER = MODEL_QUBITS
N_PARAMS_BLOCK = 4
N_PARAMS_LAYER_BLOCKS = (MODEL_QUBITS - 1) * DEPTH * N_PARAMS_BLOCK
N_PARAMS_ALPHA = 2 * (DEPTH + 1) if USE_ANGLE_ENCODING else 0
N_PARAMS_NETWORK = N_PARAMS_FIRST_LAYER + N_PARAMS_LAYER_BLOCKS + N_PARAMS_ALPHA

print("="*20)
print(f"Total network parameters: {N_PARAMS_NETWORK}")
print("="*20)

# Define the model and training functions
dev = qml.device("default.qubit", wires=MODEL_QUBITS)

@jax.jit
@qml.qnode(dev, interface="jax")
def _angle_encoding_circuit(network_params, x):
    # Map shuffled tokens back to true group coordinates before angle encoding
    x_group = PI_INV[x]

    params_per_layer = (MODEL_QUBITS - 1) * 4
    theta_first = network_params[:MODEL_QUBITS]
    theta_layers = network_params[MODEL_QUBITS:MODEL_QUBITS + DEPTH * params_per_layer].reshape((DEPTH, params_per_layer))
    alpha = network_params[MODEL_QUBITS + DEPTH * params_per_layer:].reshape((DEPTH + 1, 2))

    angle_x1 = 2.0 * jnp.pi * x_group[0] / dataset.p
    angle_x2 = 2.0 * jnp.pi * x_group[1] / dataset.p

    def encode_angles(scale_x1, scale_x2):
        for w in range(MODEL_QUBITS // 2):
            qml.RY(scale_x1 * angle_x1, wires=w)
            qml.RY(scale_x2 * angle_x2, wires=w + MODEL_QUBITS // 2)

    # Initial trainable single-qubit layer
    for w in range(MODEL_QUBITS):
        qml.RY(theta_first[w], wires=w)

    # Initial angle-encoding pass
    encode_angles(alpha[0, 0], alpha[0, 1])

    @qml.for_loop(0, DEPTH, 1)
    def layer_loop(i):
        layer_p = theta_layers[i]

        for j in range(MODEL_QUBITS - 1):
            p = layer_p[j*4 : (j+1)*4]
            qml.CNOT(wires=[j, j + 1])
            qml.RY(p[0], wires=j)
            qml.RY(p[1], wires=j + 1)
            qml.CNOT(wires=[j, j + 1])
            qml.RY(p[2], wires=j)
            qml.RY(p[3], wires=j + 1)

        # Data re-uploading with learnable per-layer scaling
        encode_angles(alpha[i + 1, 0], alpha[i + 1, 1])

    layer_loop()

    return qml.probs(wires=range(dataset.num_bits))

def angle_encoding_circuit(network_params, x):
    probs = _angle_encoding_circuit(network_params, x)[:N_CLASSES]
    return probs / jnp.sum(probs)

if USE_ANGLE_ENCODING:
    model = jax.vmap(angle_encoding_circuit, in_axes=(None, 0))
else:
    model = jax.vmap(basis_state_circuit, in_axes=(None, 0))

# Spectral regularization: penalizes DFT energy off the k1==k2 diagonal.
# For f(a,b) = g((a+b) mod p), all DFT energy lies on k1==k2.
_diag_mask = jnp.eye(PRIME, dtype=bool)
_class_indices = jnp.arange(PRIME, dtype=float)
_all_X = dataset.X  # all p² inputs

def spectral_off_diag_energy(params):
    """Fraction of 2D DFT energy off the k1==k2 diagonal (differentiable)."""
    probs = model(params, _all_X)                         # (p², p)
    soft_preds = probs @ _class_indices                    # (p²,) — expected class
    spectrum = jnp.abs(jnp.fft.fft2(soft_preds.reshape(PRIME, PRIME)))
    power = spectrum ** 2
    return jnp.sum(power * ~_diag_mask) / (jnp.sum(power) + 1e-10)

def period_regularization(params):
    # Penalize only the excess outside the principal interval [-pi, pi].
    excess = jnp.maximum(jnp.abs(params) - jnp.pi, 0.0)
    return jnp.sum(excess ** 2)

def loss_acc(params, batch_x, batch_y, spectral_coef, period_coef):
    probs = model(params, batch_x)
    logits = jnp.log(probs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
    loss = loss + spectral_coef * spectral_off_diag_energy(params) + period_coef * period_regularization(params)
    acc = (logits.argmax(-1) == batch_y).mean()
    return loss, acc

train_frac_values = [float(v) for v in (args.train_frac if isinstance(args.train_frac, list) else [args.train_frac])]
test_frac = args.test_frac

EPOCHS = args.epochs
BATCH_SIZE = dataset.p

# Get wandb config from environment variables
load_dotenv()
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

print(f"WandB Entity: {WANDB_ENTITY}, Project: {WANDB_PROJECT}")

if not WANDB_ENTITY or not WANDB_PROJECT:
    raise ValueError(".env must define WANDB_ENTITY and WANDB_PROJECT for WandB logging.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("plots", exist_ok=True)

sweep_name = getattr(args, "sweep_name", None) or f"qgrok_d{DEPTH}_e{EPOCHS}_q{MODEL_QUBITS}"
sweep_config = {
    "name": sweep_name,
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"value": 1e-2},
        "weight_decay": {"values": [float(v) for v in args.wd_coef]},
        "noise_scale": {"values": [float(v) for v in args.noise_scale]},
        "dropout_rate": {"values": [float(v) for v in args.dropout_rate]},
        "epochs": {"value": int(EPOCHS)},
        "batch_size": {"value": int(BATCH_SIZE)},
        "depth": {"value": int(DEPTH)},
        "n_qubits": {"value": int(MODEL_QUBITS)},
        "seed": {"values": [int(SEED) + i for i in range(args.runs_per_config)]},
        "train_frac": {"values": train_frac_values},
        "spectral_reg_coef": {"values": [float(v) for v in args.spectral_reg_coef]},
        "prime": {"value": int(PRIME)},
        "full_batch_val": {"value": bool(args.full_batch_val)},
        "period_scale": {"values": [float(v) for v in args.period_scale]},
        "depth": {"value": int(DEPTH)},
    },
}

sweep_id = wandb.sweep(
    sweep=sweep_config,
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
)

def run_training():
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
    cfg = run.config

    lr = float(cfg.learning_rate)
    wd_coef = float(cfg.weight_decay)
    noise_scale = float(cfg.noise_scale)
    dropout_rate = float(getattr(cfg, "dropout_rate", 0.0))
    run_epochs = int(cfg.epochs)
    seed = int(cfg.seed)
    key = jax.random.PRNGKey(seed)
    train_frac = float(cfg.train_frac)

    split_data = dataset.get_random_split(training_frac=train_frac, test_frac=test_frac)
    train_X, train_Y = split_data['training']
    val_X, val_Y = split_data['validation']

    if dropout_rate < 0.0 or dropout_rate >= 1.0:
        raise ValueError(f"dropout_rate must satisfy 0 <= rate < 1, got {dropout_rate}")

    spectral_coef = float(cfg.spectral_reg_coef)
    period_coef = float(getattr(cfg, "period_scale", 0.0))
    print(f"Regularization scales | spectral={spectral_coef}, period={period_coef}")

    # Optimizer and train-step are built per run so sweep params are correctly bound.
    steps_per_epoch = train_X.shape[0] // BATCH_SIZE

    if USE_ANGLE_ENCODING and dropout_rate > 0.0:
        print("dropout_rate > 0 was provided, but dropout is basis-only; using dropout_rate=0 for this run.")
    effective_dropout_rate = dropout_rate if not USE_ANGLE_ENCODING else 0.0
    print(f"Basis-only dropout rate: {effective_dropout_rate:.3f}")

    # Parse weight decay schedule if provided
    wd_schedule_raw = getattr(cfg, "wd_schedule", None) or (args.wd_schedule if hasattr(args, "wd_schedule") else None)
    if wd_schedule_raw is not None:
        if isinstance(wd_schedule_raw, str):
            wd_schedule_dict = {int(k): float(v) for k, v in json.loads(wd_schedule_raw).items()}
        else:
            wd_schedule_dict = {int(k): float(v) for k, v in wd_schedule_raw.items()}
        # Sort by epoch and convert to step boundaries
        sorted_epochs = sorted(wd_schedule_dict.keys())
        schedules = [optax.constant_schedule(wd_schedule_dict[e]) for e in sorted_epochs]
        boundaries = [e * steps_per_epoch for e in sorted_epochs[1:]]
        wd_schedule = optax.join_schedules(schedules=schedules, boundaries=boundaries)
        print(f"Using weight decay schedule: { {e: wd_schedule_dict[e] for e in sorted_epochs} }")
    else:
        wd_schedule = wd_coef

    opt = optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=wd_schedule)
    spectral_coef_arr = jnp.asarray(spectral_coef, dtype=jnp.float32)
    period_coef_arr = jnp.asarray(period_coef, dtype=jnp.float32)

    def make_batches(X, y, batch_size, batch_key):
        n = X.shape[0]
        n_full = n // batch_size
        perm = jax.random.permutation(batch_key, n)[: n_full * batch_size]
        Xb = X[perm].reshape(n_full, batch_size, *X.shape[1:])
        yb = y[perm].reshape(n_full, batch_size)
        return Xb, yb

    @jax.jit
    def train_epoch(params, opt_state, Xb, yb, noise_key, spectral_coef_local, period_coef_local):
        def step(carry, batch):
            p, o, nk = carry
            bx, by = batch
            nk, mask_key, noise_subkey = jax.random.split(nk, 3)
            dropout_mask = jax.random.bernoulli(mask_key, 1.0 - effective_dropout_rate, shape=p.shape).astype(p.dtype)
            dropped_p = p * dropout_mask
            noisy_p = dropped_p + noise_scale * jax.random.normal(noise_subkey, p.shape)
            (loss, acc), grads = jax.value_and_grad(loss_acc, has_aux=True)(
                noisy_p,
                bx,
                by,
                spectral_coef_local,
                period_coef_local,
            )
            # Update the clean params, not the noisy ones
            updates, o = opt.update(grads, o, p)
            p = optax.apply_updates(p, updates)
            return (p, o, nk), (loss, acc)

        (params, opt_state, _), (losses, accs) = jax.lax.scan(step, (params, opt_state, noise_key), (Xb, yb))
        return params, opt_state, jnp.mean(losses), jnp.mean(accs)

    @jax.jit
    def eval_epoch(params, Xb, yb, spectral_coef_local, period_coef_local):
        def step(_, batch):
            bx, by = batch
            loss, acc = loss_acc(params, bx, by, spectral_coef_local, period_coef_local)
            return None, (loss, acc)

        _, (losses, accs) = jax.lax.scan(step, None, (Xb, yb))
        return jnp.mean(losses), jnp.mean(accs)

    # params = jax.random.normal(key, (N_PARAMS_NETWORK,)) * jnp.sqrt(2 / N_PARAMS_NETWORK) #Kaiming init
    # old: ignoring kaiming init, just do a regular uniform init for angle parameters:
    params = jax.random.uniform(key, (N_PARAMS_NETWORK,), minval=-jnp.pi, maxval=jnp.pi)
    opt_state = opt.init(params)

    train_loss_curve, val_loss_curve = [], []
    train_acc_curve, val_acc_curve = [], []
    dft_snapshots = []  # list of (epoch, pred_spectrum)

    run_key = key

    for epoch in range(1, run_epochs + 1):
        run_key, train_key, val_key, noise_key = jax.random.split(run_key, 4)
        train_Xb, train_yb = make_batches(train_X, train_Y, BATCH_SIZE, train_key)

        if cfg.full_batch_val:
            # Keep a leading batch axis so eval_epoch sees one full-size batch.
            val_Xb = jnp.expand_dims(val_X, axis=0)
            val_yb = jnp.expand_dims(val_Y, axis=0)
        else:
            val_Xb, val_yb = make_batches(val_X, val_Y, BATCH_SIZE, val_key)
        


        params, opt_state, tl, ta = train_epoch(
            params,
            opt_state,
            train_Xb,
            train_yb,
            noise_key,
            spectral_coef_arr,
            period_coef_arr,
        )
        vl, va = eval_epoch(params, val_Xb, val_yb, spectral_coef_arr, period_coef_arr)
        l2_norm = float(jnp.sqrt(jnp.sum(params ** 2)))
        param_max_abs = float(jnp.max(jnp.abs(params)))
        period_reg = float(period_regularization(params))
        current_wd = float(opt_state.hyperparams['weight_decay'])

        train_loss_curve.append(tl)
        val_loss_curve.append(vl)
        train_acc_curve.append(ta)
        val_acc_curve.append(va)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": float(tl),
                "val_loss": float(vl),
                "train_acc": float(ta),
                "val_acc": float(va),
                "param_l2_norm": l2_norm,
                "param_max_abs": param_max_abs,
                "weight_decay": current_wd,
                "noise_scale": noise_scale,
                "dropout_rate": effective_dropout_rate,
                "spectral_reg_coef": spectral_coef,
                "period_scale": period_coef,
                "period_reg_raw": period_reg,
                "period_reg_term": period_coef * period_reg,
            }
        )

        if epoch % max(1, run_epochs // 10) == 0:
            off_diag = float(spectral_off_diag_energy(params))
            wandb.log({"spectral_off_diag": off_diag, "epoch": epoch})
            print(f"Epoch {epoch:03d}/{run_epochs} | "
                f"train_loss={float(tl):.4f}, val_loss={float(vl):.4f}, "
                f"train_acc={float(ta):.4f}, val_acc={float(va):.4f}, "
                f"l2_norm={l2_norm:.4f}, max|param|={param_max_abs:.4f}, "
                f"wd={current_wd}, period_reg={period_reg:.4e}, spectral_off_diag={off_diag:.4f}")
            
            # Save latest checkpoint as a WandB artifact.
            artifact = wandb.Artifact(
                name=f"latest-model-depth-{DEPTH}-epochs-{run_epochs}-wd-{wd_coef}",
                type="model",
                metadata={"epoch": epoch, "val_loss": float(vl), "weight_decay": wd_coef, "noise_scale": noise_scale},
            )

            flat_opt, _opt_tree_def = jax.tree.flatten(opt_state)
            save_dict = {
                "params": numpy.asarray(params),
                "epoch": epoch,
                "val_loss": float(vl),
                "weight_decay": wd_coef,
                "noise_scale": noise_scale,
            }
            for i, leaf in enumerate(flat_opt):
                save_dict[f"opt_{i}"] = numpy.asarray(leaf)
            save_dict["opt_n_leaves"] = len(flat_opt)

            with artifact.new_file("latest_model.npz", mode="wb") as f:
                numpy.savez(f, **save_dict)
            run.log_artifact(artifact, aliases=["latest"])

            # Record DFT snapshot of model predictions
            snap_probs = model(params, dataset.X)
            snap_preds = numpy.array(jnp.argmax(snap_probs, axis=-1)).reshape(PRIME, PRIME).astype(float)
            dft_snapshots.append((epoch, numpy.abs(numpy.fft.fft2(snap_preds))))

        elif epoch == 1:
            print(f"Epoch {epoch}/{run_epochs} has completed.")
            print(f"Epoch {epoch:03d}/{run_epochs} | "
                f"train_loss={float(tl):.4f}, val_loss={float(vl):.4f}, "
                f"train_acc={float(ta):.4f}, val_acc={float(va):.4f}, "
                f"l2_norm={l2_norm:.4f}, wd={current_wd}, noise={noise_scale}")

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
    ax[0].plot(train_loss_curve, label="Train")
    ax[0].plot(val_loss_curve, label="Validation")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(train_acc_curve)
    ax[1].plot(val_acc_curve)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")

    plt.tight_layout()
    plot_path = f"plots/training_curves_epochs_{run_epochs}_depth_{DEPTH}_wd_{wd_coef}_noise_{noise_scale}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {plot_path}")

    wandb.log({"training_curves": wandb.Image(plot_path)})

    # --- DFT evolution: ground truth spectrum + snapshots over training ---
    truth_grid = numpy.array(dataset.Y).reshape(PRIME, PRIME).astype(float)
    truth_spectrum = numpy.abs(numpy.fft.fft2(truth_grid))

    n_snaps = len(dft_snapshots)
    fig_dft, axes_dft = plt.subplots(1, n_snaps + 1, figsize=(3 * (n_snaps + 1), 3))
    if n_snaps + 1 == 1:
        axes_dft = [axes_dft]

    # First panel: ground truth
    im = axes_dft[0].imshow(truth_spectrum, cmap="viridis")
    axes_dft[0].set_title("Truth")
    fig_dft.colorbar(im, ax=axes_dft[0], fraction=0.046)

    # Remaining panels: model DFT at each snapshot epoch
    for idx, (snap_epoch, snap_spectrum) in enumerate(dft_snapshots):
        diff_mse = float(numpy.mean((truth_spectrum - snap_spectrum) ** 2))
        im = axes_dft[idx + 1].imshow(snap_spectrum, cmap="viridis")
        axes_dft[idx + 1].set_title(f"Ep {snap_epoch}\nMSE={diff_mse:.1f}")
        fig_dft.colorbar(im, ax=axes_dft[idx + 1], fraction=0.046)

    # Use class-formatted ticks [k] on both axes with identical integer spacing.
    class_ticks = list(range(PRIME))
    class_labels = [f"[{k}]" for k in class_ticks]
    for ax_dft in axes_dft:
        ax_dft.set_xticks(class_ticks)
        ax_dft.set_yticks(class_ticks)
        ax_dft.set_xticklabels(class_labels)
        ax_dft.set_yticklabels(class_labels)

    plt.tight_layout()
    dft_plot_path = f"plots/dft_evolution_epochs_{run_epochs}_depth_{DEPTH}_wd_{wd_coef}.png"
    plt.savefig(dft_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    final_mse = float(numpy.mean((truth_spectrum - dft_snapshots[-1][1]) ** 2)) if dft_snapshots else float("nan")
    wandb.log({
        "dft_evolution": wandb.Image(dft_plot_path),
        "dft_spectrum_mse": final_mse,
    })
    print(f"DFT evolution saved to {dft_plot_path} | final spectrum MSE: {final_mse:.4f}")

    wandb.finish()

print(f"Created W&B sweep: {sweep_id}")
wandb.agent(
    sweep_id=sweep_id,
    function=run_training,
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    count=args.sweep_count,
)


