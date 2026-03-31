"""Resume training from a wandb checkpoint with modified weight decay."""

import os
import operator
import json
import pennylane as qml
from pennylane import numpy as np
from cayley_table import CayleyTableDataset
import jax
from jax import numpy as jnp
import numpy
import optax
import wandb
import argparse
from dotenv import load_dotenv
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Resume quantum grokking training from a W&B checkpoint.")


def parse_optional_bool(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


parser.add_argument("--run_id", type=str, required=True, help="W&B run ID to load checkpoint from")
parser.add_argument("--epochs", type=int, default=100, help="Additional epochs to train (default: 100)")
parser.add_argument("--wd_coef", type=float, default=None, nargs="+", help="New weight decay values for sweep (default: source run value)")
parser.add_argument("--wd_schedule", type=str, default=None,
                    help='JSON dict mapping epoch->weight_decay, e.g. \'{"0":0.01,"100":0.001}\'. Overrides --wd_coef.')
parser.add_argument("--sweep_count", type=int, default=None, help="Maximum number of sweep runs")
parser.add_argument("--learning_rate", type=float, default=None, nargs="+", help="Learning rate values for sweep (default: source run value)")
parser.add_argument("--fresh_optimizer", action="store_true", help="Reinitialize optimizer state instead of restoring from checkpoint")
parser.add_argument("--train_frac", type=float, default=None, help="Fraction of data for training (default: source run value)")
parser.add_argument("--test_frac", type=float, default=None, help="Fraction of data for testing (default: source run value)")
parser.add_argument("--noise_scale", type=float, default=None, nargs="+", help="Parameter noise scale values (default: source run value)")
parser.add_argument("--dropout_rate", type=float, default=None, nargs="+", help="Basis-state dropout rates (default: source run value)")
parser.add_argument("--spectral_reg_coef", type=float, default=None, nargs="+", help="Spectral regularization coefficients (default: source run value)")
parser.add_argument("--period_scale", type=float, default=None, nargs="+", help="Period regularization scales (default: source run value)")
parser.add_argument("--angle_encoding", type=parse_optional_bool, nargs="?", const=True, default=None,
                    help="Set angle encoding explicitly (true/false). If omitted, infer from checkpoint parameter count.")
parser.add_argument("--prime", type=int, default=None, help="Override prime p. Defaults to source run config when available.")
parser.add_argument("--depth", type=int, default=None, help="Override circuit depth. Defaults to source run config when available.")
parser.add_argument("--full_batch_val", type=parse_optional_bool, nargs="?", const=True, default=None,
                    help="Use full validation batch (default: source run config)")
parser.add_argument("--resume_original_run", action="store_true",
                    help="Resume and log directly into the original W&B run id (single-config mode, no sweep).")
args = parser.parse_args()

# ── W&B setup ─────────────────────────────────────────────────────────────────

load_dotenv()
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

if not WANDB_ENTITY or not WANDB_PROJECT:
    raise ValueError(".env must define WANDB_ENTITY and WANDB_PROJECT.")


def get_source_run(run_id):
    api = wandb.Api()
    return api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")


source_run = get_source_run(args.run_id)
source_cfg = source_run.config or {}


def source_or_default(key, default):
    val = source_cfg.get(key, default)
    return default if val is None else val


def resolve_sweep_values(arg_values, source_key, default_value):
    if arg_values is not None:
        return [float(v) for v in arg_values]
    return [float(source_or_default(source_key, default_value))]


LEARNING_RATE_VALUES = resolve_sweep_values(args.learning_rate, "learning_rate", 1e-2)
WD_VALUES = resolve_sweep_values(args.wd_coef, "weight_decay", 1e-3)
NOISE_VALUES = resolve_sweep_values(args.noise_scale, "noise_scale", 0.0)
DROPOUT_VALUES = resolve_sweep_values(args.dropout_rate, "dropout_rate", 0.0)
SPECTRAL_VALUES = resolve_sweep_values(args.spectral_reg_coef, "spectral_reg_coef", 0.0)
PERIOD_VALUES = resolve_sweep_values(args.period_scale, "period_scale", 0.0)
TRAIN_FRAC = float(args.train_frac) if args.train_frac is not None else float(source_or_default("train_frac", 0.8))
TEST_FRAC = float(args.test_frac) if args.test_frac is not None else float(source_or_default("test_frac", 0.0))
FULL_BATCH_VAL = bool(args.full_batch_val) if args.full_batch_val is not None else bool(source_or_default("full_batch_val", False))
WD_SCHEDULE = args.wd_schedule if args.wd_schedule is not None else source_cfg.get("wd_schedule", None)

# ── Dataset ───────────────────────────────────────────────────────────────────

PRIME = int(args.prime) if args.prime is not None else int(source_cfg.get("prime", 13))
dataset = CayleyTableDataset(p=PRIME, seed=42, op=operator.add)

SEED = int(source_cfg.get("seed", 42))
MODEL_QUBITS = dataset.num_bits * 2
DEPTH = int(args.depth) if args.depth is not None else int(source_cfg.get("depth", 30))
N_CLASSES = dataset.p

# ── Circuit config ────────────────────────────────────────────────────────────

def make_circuit_config(n_qubits, depth, angle_encoding=False):
    n_first_layer = n_qubits
    n_block = 4
    n_layer_blocks = (n_qubits - 1) * depth * n_block
    n_alpha = 2 * (depth + 1) if angle_encoding else 0
    n_network = n_first_layer + n_layer_blocks + n_alpha
    return SimpleNamespace(
        n_qubits=n_qubits, depth=depth,
        N_PARAMS_FIRST_LAYER=n_first_layer, N_PARAMS_BLOCK=n_block,
        N_PARAMS_LAYER_BLOCKS=n_layer_blocks, N_PARAMS_ALPHA=n_alpha,
        N_PARAMS_NETWORK=n_network,
    )

key = jax.random.PRNGKey(SEED)
dev = qml.device("default.qubit", wires=MODEL_QUBITS)

# ── Circuits ──────────────────────────────────────────────────────────────────

@jax.jit
@qml.qnode(dev, interface="jax")
def _basis_state_circuit(network_params, x):
    params = iter(network_params)
    qml.BasisState(x[0], wires=range(MODEL_QUBITS // 2))
    qml.BasisState(x[1], wires=range(MODEL_QUBITS // 2, MODEL_QUBITS))
    for w in range(MODEL_QUBITS):
        qml.RY(next(params), wires=w)
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

PI_INV = jnp.argsort(dataset.pi)

@jax.jit
@qml.qnode(dev, interface="jax")
def _angle_encoding_circuit(network_params, x):
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

    for w in range(MODEL_QUBITS):
        qml.RY(theta_first[w], wires=w)
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
        encode_angles(alpha[i + 1, 0], alpha[i + 1, 1])
    layer_loop()
    return qml.probs(wires=range(dataset.num_bits))

def angle_encoding_circuit(network_params, x):
    probs = _angle_encoding_circuit(network_params, x)[:N_CLASSES]
    return probs / jnp.sum(probs)


def infer_angle_encoding_from_params(param_count):
    basis_n = make_circuit_config(MODEL_QUBITS, DEPTH, angle_encoding=False).N_PARAMS_NETWORK
    angle_n = make_circuit_config(MODEL_QUBITS, DEPTH, angle_encoding=True).N_PARAMS_NETWORK
    if param_count == basis_n:
        return False
    if param_count == angle_n:
        return True
    raise ValueError(
        f"Checkpoint parameter count {param_count} is incompatible with this resume configuration "
        f"(basis={basis_n}, angle={angle_n}, depth={DEPTH}, prime={PRIME})."
    )

# ── Data split ────────────────────────────────────────────────────────────────

split_data = dataset.get_random_split(training_frac=TRAIN_FRAC, test_frac=TEST_FRAC)
train_X, train_Y = split_data['training']
val_X, val_Y = split_data['validation']
BATCH_SIZE = dataset.p

os.makedirs("plots", exist_ok=True)

# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(run_id):
    """Download the latest model artifact from a W&B run and return the npz data."""
    run = get_source_run(run_id)
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if not artifacts:
        raise ValueError(f"No model artifacts found for run {run_id}")
    # Get the most recent artifact
    artifact = artifacts[-1]
    print(f"Loading artifact: {artifact.name} (epoch {artifact.metadata.get('epoch', '?')})")
    artifact_dir = artifact.download()
    data = numpy.load(os.path.join(artifact_dir, "latest_model.npz"), allow_pickle=True)
    return data

# ── Sweep config ──────────────────────────────────────────────────────────────

sweep_config = {
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": LEARNING_RATE_VALUES},
        "weight_decay": {"values": WD_VALUES},
        "noise_scale": {"values": NOISE_VALUES},
        "dropout_rate": {"values": DROPOUT_VALUES},
        "spectral_reg_coef": {"values": SPECTRAL_VALUES},
        "period_scale": {"values": PERIOD_VALUES},
        "epochs": {"value": int(args.epochs)},
        "batch_size": {"value": int(BATCH_SIZE)},
        "depth": {"value": int(DEPTH)},
        "prime": {"value": int(PRIME)},
        "n_qubits": {"value": int(MODEL_QUBITS)},
        "seed": {"value": int(SEED)},
        "train_frac": {"value": float(TRAIN_FRAC)},
        "test_frac": {"value": float(TEST_FRAC)},
        "wd_schedule": {"value": WD_SCHEDULE},
        "full_batch_val": {"value": bool(FULL_BATCH_VAL)},
        "source_run_id": {"value": args.run_id},
        "fresh_optimizer": {"value": args.fresh_optimizer},
    },
}

# ── Training ──────────────────────────────────────────────────────────────────

def run_training(training_cfg=None, resume_original=False):
    if resume_original:
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, id=args.run_id, resume="must")
        if training_cfg is not None:
            run.config.update(training_cfg, allow_val_change=True)
    else:
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
    try:
        _run_training_inner(run, training_cfg=training_cfg)
    except KeyboardInterrupt:
        print("\nInterrupted — skipping to next sweep run.")
        wandb.finish()

def _run_training_inner(run, training_cfg=None):
    cfg = training_cfg if training_cfg is not None else run.config

    def cfg_get(name, default=None):
        if isinstance(cfg, dict):
            return cfg.get(name, default)
        return getattr(cfg, name, default)

    lr = float(cfg_get("learning_rate"))
    wd_coef = float(cfg_get("weight_decay"))
    noise_scale = float(cfg_get("noise_scale", 0.0))
    dropout_rate = float(cfg_get("dropout_rate", 0.0))
    spectral_coef = float(cfg_get("spectral_reg_coef", 0.0))
    period_coef = float(cfg_get("period_scale", 0.0))
    run_epochs = int(cfg_get("epochs"))
    steps_per_epoch = train_X.shape[0] // BATCH_SIZE
    # W&B run step can be ahead of checkpoint epoch when resuming an existing run.
    # Track a monotonic step counter for wandb.log(step=...).
    wandb_step = int(getattr(run, "step", 0))

    # Load checkpoint
    ckpt = load_checkpoint(args.run_id)
    params = jnp.array(ckpt["params"])
    param_count = int(params.shape[0])
    inferred_angle_encoding = infer_angle_encoding_from_params(param_count)
    if args.angle_encoding is not None and bool(args.angle_encoding) != inferred_angle_encoding:
        raise ValueError(
            f"--angle_encoding={args.angle_encoding} conflicts with checkpoint parameter count {param_count}. "
            f"Inferred angle_encoding={inferred_angle_encoding}."
        )
    use_angle_encoding = inferred_angle_encoding if args.angle_encoding is None else bool(args.angle_encoding)
    model = jax.vmap(angle_encoding_circuit, in_axes=(None, 0)) if use_angle_encoding else jax.vmap(basis_state_circuit, in_axes=(None, 0))
    if dropout_rate < 0.0 or dropout_rate >= 1.0:
        raise ValueError(f"dropout_rate must satisfy 0 <= rate < 1, got {dropout_rate}")
    if use_angle_encoding and dropout_rate > 0.0:
        print("dropout_rate > 0 was provided, but dropout is basis-only; using dropout_rate=0 for this run.")
    effective_dropout_rate = dropout_rate if not use_angle_encoding else 0.0

    diag_mask = jnp.eye(PRIME, dtype=bool)
    class_indices = jnp.arange(PRIME, dtype=float)
    all_X = dataset.X

    start_epoch = int(ckpt["epoch"])
    old_wd = float(ckpt["weight_decay"])
    n_opt_leaves = int(ckpt["opt_n_leaves"])
    saved_leaves = [jnp.array(ckpt[f"opt_{i}"]) for i in range(n_opt_leaves)]

    print(
        f"Resuming from epoch {start_epoch}, old WD={old_wd}, new WD={wd_coef}, "
        f"prime={PRIME}, depth={DEPTH}, angle_encoding={use_angle_encoding}, "
        f"noise_scale={noise_scale}, dropout_rate={effective_dropout_rate}, "
        f"spectral_reg_coef={spectral_coef}, period_scale={period_coef}"
    )

    def spectral_off_diag_energy(params_local):
        probs = model(params_local, all_X)
        soft_preds = probs @ class_indices
        spectrum = jnp.abs(jnp.fft.fft2(soft_preds.reshape(PRIME, PRIME)))
        power = spectrum ** 2
        return jnp.sum(power * ~diag_mask) / (jnp.sum(power) + 1e-10)

    def period_regularization(params_local):
        excess = jnp.maximum(jnp.abs(params_local) - jnp.pi, 0.0)
        return jnp.sum(excess ** 2)

    def loss_acc(params_local, batch_x, batch_y):
        probs = model(params_local, batch_x)
        logits = jnp.log(probs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
        loss = loss + spectral_coef * spectral_off_diag_energy(params_local) + period_coef * period_regularization(params_local)
        acc = (logits.argmax(-1) == batch_y).mean()
        return loss, acc

    # Build optimizer with new weight decay
    wd_schedule_raw = cfg_get("wd_schedule", None) or WD_SCHEDULE
    if wd_schedule_raw is not None:
        if isinstance(wd_schedule_raw, str):
            wd_schedule_dict = {int(k): float(v) for k, v in json.loads(wd_schedule_raw).items()}
        else:
            wd_schedule_dict = {int(k): float(v) for k, v in wd_schedule_raw.items()}
        sorted_epochs = sorted(wd_schedule_dict.keys())
        schedules = [optax.constant_schedule(wd_schedule_dict[e]) for e in sorted_epochs]
        boundaries = [e * steps_per_epoch for e in sorted_epochs[1:]]
        wd_schedule = optax.join_schedules(schedules=schedules, boundaries=boundaries)
        print(f"Using weight decay schedule: { {e: wd_schedule_dict[e] for e in sorted_epochs} }")
    else:
        wd_schedule = wd_coef

    opt = optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=wd_schedule)

    if args.fresh_optimizer:
        print("Using fresh optimizer state.")
        opt_state = opt.init(params)
    else:
        # Restore optimizer state: init fresh to get tree structure, then swap in saved leaves
        fresh_state = opt.init(params)
        _, tree_def = jax.tree.flatten(fresh_state)
        opt_state = jax.tree.unflatten(tree_def, saved_leaves)
        # Override hyperparams with the new values
        opt_state.hyperparams['weight_decay'] = fresh_state.hyperparams['weight_decay']
        opt_state.hyperparams['learning_rate'] = jnp.array(lr, dtype=jnp.float32)

    # ── Training helpers ──────────────────────────────────────────────────────

    def make_batches(X, y, batch_size, batch_key):
        n = X.shape[0]
        n_full = n // batch_size
        perm = jax.random.permutation(batch_key, n)[: n_full * batch_size]
        Xb = X[perm].reshape(n_full, batch_size, *X.shape[1:])
        yb = y[perm].reshape(n_full, batch_size)
        return Xb, yb

    @jax.jit
    def train_epoch(params, opt_state, Xb, yb, noise_key):
        def step(carry, batch):
            p, o, nk = carry
            bx, by = batch
            nk, mask_key, noise_subkey = jax.random.split(nk, 3)
            dropout_mask = jax.random.bernoulli(mask_key, 1.0 - effective_dropout_rate, shape=p.shape).astype(p.dtype)
            dropped_p = p * dropout_mask
            noisy_p = dropped_p + noise_scale * jax.random.normal(noise_subkey, p.shape)
            (loss, acc), grads = jax.value_and_grad(loss_acc, has_aux=True)(noisy_p, bx, by)
            updates, o = opt.update(grads, o, p)
            p = optax.apply_updates(p, updates)
            return (p, o, nk), (loss, acc)
        (params, opt_state, _), (losses, accs) = jax.lax.scan(step, (params, opt_state, noise_key), (Xb, yb))
        return params, opt_state, jnp.mean(losses), jnp.mean(accs)

    @jax.jit
    def eval_epoch(params, Xb, yb):
        def step(_, batch):
            bx, by = batch
            loss, acc = loss_acc(params, bx, by)
            return None, (loss, acc)
        _, (losses, accs) = jax.lax.scan(step, None, (Xb, yb))
        return jnp.mean(losses), jnp.mean(accs)

    # ── Training loop ─────────────────────────────────────────────────────────

    train_loss_curve, val_loss_curve = [], []
    train_acc_curve, val_acc_curve = [], []
    dft_snapshots = []

    run_key = key
    for epoch in range(start_epoch + 1, start_epoch + run_epochs + 1):
        run_key, train_key, val_key, noise_key = jax.random.split(run_key, 4)
        train_Xb, train_yb = make_batches(train_X, train_Y, BATCH_SIZE, train_key)
        if bool(cfg_get("full_batch_val", False)):
            val_Xb = jnp.expand_dims(val_X, axis=0)
            val_yb = jnp.expand_dims(val_Y, axis=0)
        else:
            val_Xb, val_yb = make_batches(val_X, val_Y, BATCH_SIZE, val_key)

        params, opt_state, tl, ta = train_epoch(params, opt_state, train_Xb, train_yb, noise_key)
        vl, va = eval_epoch(params, val_Xb, val_yb)
        l2_norm = float(jnp.sqrt(jnp.sum(params ** 2)))
        current_wd = float(opt_state.hyperparams['weight_decay'])

        train_loss_curve.append(tl)
        val_loss_curve.append(vl)
        train_acc_curve.append(ta)
        val_acc_curve.append(va)

        wandb_step = max(wandb_step + 1, epoch)
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": float(tl),
                "val_loss": float(vl),
                "train_acc": float(ta),
                "val_acc": float(va),
                "param_l2_norm": l2_norm,
                "weight_decay": current_wd,
                "noise_scale": noise_scale,
                "dropout_rate": effective_dropout_rate,
            },
            step=wandb_step,
        )

        if epoch % max(1, run_epochs // 10) == 0:
            off_diag = float(spectral_off_diag_energy(params))
            wandb.log({"spectral_off_diag": off_diag, "epoch": epoch}, step=wandb_step)
            print(f"Epoch {epoch:03d}/{start_epoch + run_epochs} | "
                  f"train_loss={float(tl):.4f}, val_loss={float(vl):.4f}, "
                  f"train_acc={float(ta):.4f}, val_acc={float(va):.4f}, "
                  f"l2_norm={l2_norm:.4f}, wd={current_wd}, spectral_off_diag={off_diag:.4f}")

            artifact = wandb.Artifact(
                name=f"resumed-model-depth-{DEPTH}-epochs-{epoch}-wd-{wd_coef}",
                type="model",
                metadata={"epoch": epoch, "val_loss": float(vl), "weight_decay": wd_coef,
                           "source_run_id": args.run_id, "noise_scale": noise_scale},
            )
            flat_opt, _ = jax.tree.flatten(opt_state)
            save_dict = {
                "params": numpy.asarray(params),
                "epoch": epoch,
                "val_loss": float(vl),
                "weight_decay": wd_coef,
            }
            for i, leaf in enumerate(flat_opt):
                save_dict[f"opt_{i}"] = numpy.asarray(leaf)
            save_dict["opt_n_leaves"] = len(flat_opt)

            with artifact.new_file("latest_model.npz", mode="wb") as f:
                numpy.savez(f, **save_dict)
            run.log_artifact(artifact, aliases=["latest"])

            snap_probs = model(params, dataset.X)
            snap_preds = numpy.array(jnp.argmax(snap_probs, axis=-1)).reshape(PRIME, PRIME).astype(float)
            dft_snapshots.append((epoch, numpy.abs(numpy.fft.fft2(snap_preds))))

        elif epoch == start_epoch + 1:
            print(f"Epoch {epoch}/{start_epoch + run_epochs} | "
                  f"train_loss={float(tl):.4f}, val_loss={float(vl):.4f}, "
                  f"train_acc={float(ta):.4f}, val_acc={float(va):.4f}, "
                  f"l2_norm={l2_norm:.4f}, wd={current_wd}")

    # ── Plot ──────────────────────────────────────────────────────────────────

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
    plot_path = f"plots/resumed_curves_epochs_{start_epoch}-{start_epoch + run_epochs}_depth_{DEPTH}_wd_{wd_coef}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {plot_path}")

    wandb.log({"training_curves": wandb.Image(plot_path)})

    truth_grid = numpy.array(dataset.Y).reshape(PRIME, PRIME).astype(float)
    truth_spectrum = numpy.abs(numpy.fft.fft2(truth_grid))

    n_snaps = len(dft_snapshots)
    fig_dft, axes_dft = plt.subplots(1, n_snaps + 1, figsize=(3 * (n_snaps + 1), 3))
    if n_snaps + 1 == 1:
        axes_dft = [axes_dft]

    im = axes_dft[0].imshow(truth_spectrum, cmap="viridis")
    axes_dft[0].set_title("Truth")
    fig_dft.colorbar(im, ax=axes_dft[0], fraction=0.046)

    for idx, (snap_epoch, snap_spectrum) in enumerate(dft_snapshots):
        diff_mse = float(numpy.mean((truth_spectrum - snap_spectrum) ** 2))
        im = axes_dft[idx + 1].imshow(snap_spectrum, cmap="viridis")
        axes_dft[idx + 1].set_title(f"Ep {snap_epoch}\\nMSE={diff_mse:.1f}")
        fig_dft.colorbar(im, ax=axes_dft[idx + 1], fraction=0.046)

    class_ticks = list(range(PRIME))
    class_labels = [f"[{k}]" for k in class_ticks]
    for ax_dft in axes_dft:
        ax_dft.set_xticks(class_ticks)
        ax_dft.set_yticks(class_ticks)
        ax_dft.set_xticklabels(class_labels)
        ax_dft.set_yticklabels(class_labels)

    plt.tight_layout()
    dft_plot_path = f"plots/resumed_dft_evolution_epochs_{start_epoch}-{start_epoch + run_epochs}_depth_{DEPTH}_wd_{wd_coef}.png"
    plt.savefig(dft_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    final_mse = float(numpy.mean((truth_spectrum - dft_snapshots[-1][1]) ** 2)) if dft_snapshots else float("nan")
    wandb.log({
        "dft_evolution": wandb.Image(dft_plot_path),
        "dft_spectrum_mse": final_mse,
    })
    print(f"DFT evolution saved to {dft_plot_path} | final spectrum MSE: {final_mse:.4f}")

    wandb.finish()

def _single_value(name, values):
    if len(values) != 1:
        raise ValueError(
            f"--resume_original_run requires exactly one value for {name}, got {values}."
        )
    return float(values[0])


# ── Launch sweep / single resumed run ────────────────────────────────────────

if args.resume_original_run:
    one_cfg = {
        "learning_rate": _single_value("learning_rate", LEARNING_RATE_VALUES),
        "weight_decay": _single_value("weight_decay", WD_VALUES),
        "noise_scale": _single_value("noise_scale", NOISE_VALUES),
        "dropout_rate": _single_value("dropout_rate", DROPOUT_VALUES),
        "spectral_reg_coef": _single_value("spectral_reg_coef", SPECTRAL_VALUES),
        "period_scale": _single_value("period_scale", PERIOD_VALUES),
        "epochs": int(args.epochs),
        "batch_size": int(BATCH_SIZE),
        "depth": int(DEPTH),
        "prime": int(PRIME),
        "n_qubits": int(MODEL_QUBITS),
        "seed": int(SEED),
        "train_frac": float(TRAIN_FRAC),
        "test_frac": float(TEST_FRAC),
        "wd_schedule": WD_SCHEDULE,
        "full_batch_val": bool(FULL_BATCH_VAL),
        "source_run_id": args.run_id,
        "fresh_optimizer": bool(args.fresh_optimizer),
    }
    print(f"Resuming original W&B run id: {args.run_id}")
    run_training(training_cfg=one_cfg, resume_original=True)
else:
    sweep_id = wandb.sweep(sweep=sweep_config, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    print(f"Created W&B sweep: {sweep_id}")
    wandb.agent(
        sweep_id=sweep_id,
        function=run_training,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        count=args.sweep_count,
    )
