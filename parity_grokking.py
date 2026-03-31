#!/usr/bin/env python
# coding: utf-8

# Imports

# In[2]:


import os
import math
from enum import Enum
import pennylane as qml
import jax
from jax import numpy as jnp
import numpy
import optax
import wandb
import argparse
from dotenv import load_dotenv
from parity_dataset import ParityDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Argument parsing

# In[ ]:


defaults = {
    "n_bits": 10,
    "depth": 20, # SHOULD BE AN EVEN NUMBER!
    "epochs": 100,
    "wd_coef": [1e-4, 1e-3, 1e-2],
    "sweep_count": None,
    "train_frac": [0.8],
    "test_frac": 0.0,
    "seed": 42,
    "encoding": "basis",  # "basis", "angle", or "amplitude"
}

parser = argparse.ArgumentParser(description="Parity grokking with variational quantum classifier.")
parser.add_argument("--n_bits", type=int, default=defaults["n_bits"], help="Length of bitstrings")
parser.add_argument("--depth", type=int, default=defaults["depth"], help="Number of SO(4) block layers")
parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="Number of training epochs")
parser.add_argument("--wd_coef", type=float, nargs="+", default=defaults["wd_coef"], help="Weight decay values for W&B sweep")
parser.add_argument("--sweep_count", type=int, default=defaults["sweep_count"], help="Max number of sweep runs")
parser.add_argument("--train_frac", type=float, nargs="+", default=defaults["train_frac"], help="Training fraction values for W&B sweep")
parser.add_argument("--test_frac", type=float, default=defaults["test_frac"], help="Fraction of data for testing")
parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed")
parser.add_argument("--encoding", type=str, choices=["basis", "angle", "amplitude"], default=defaults["encoding"], help="Encoding type: basis, angle, or amplitude")
args = parser.parse_args()


# Dataset

# In[ ]:


dataset = ParityDataset(n=args.n_bits, seed=args.seed)
print(f"Parity dataset: n={dataset.n}, total samples={dataset.num_samples}")


# Circuit

# In[ ]:


from types import SimpleNamespace


class EncodingType(Enum):
    BASIS = "basis"
    ANGLE = "angle"
    AMPLITUDE = "amplitude"


def get_n_qubits(encoding: EncodingType, n_bits: int) -> int:
    """Return the number of qubits required for the given encoding."""
    if encoding == EncodingType.AMPLITUDE:
        return math.ceil(math.log2(n_bits))
    return n_bits


def make_circuit_config(n_qubits: int, depth: int, encoding: EncodingType):
    """Compute parameter counts for each circuit variant."""
    n_first_layer = n_qubits
    n_block = 4
    n_layer_blocks = (n_qubits - 1) * depth * n_block
    n_alpha = (depth + 1) if encoding == EncodingType.ANGLE else 0
    n_network = n_first_layer + n_layer_blocks + n_alpha
    return SimpleNamespace(
        n_qubits=n_qubits,
        depth=depth,
        encoding=encoding,
        N_PARAMS_FIRST_LAYER=n_first_layer,
        N_PARAMS_BLOCK=n_block,
        N_PARAMS_LAYER_BLOCKS=n_layer_blocks,
        N_PARAMS_ALPHA=n_alpha,
        N_PARAMS_NETWORK=n_network,
    )


def _make_basis_circuit(n_qubits, depth, dev):
    @jax.jit
    @qml.qnode(dev, interface="jax")
    def circuit(network_params, x):
        params = iter(network_params)
        qml.BasisState(x, wires=range(n_qubits))
        for w in range(n_qubits):
            qml.RY(next(params), wires=w)
        for _ in range(depth):
            for j in range(n_qubits - 1):
                qml.CNOT(wires=[j, j + 1])
                qml.RY(next(params), wires=j)
                qml.RY(next(params), wires=j + 1)
                qml.CNOT(wires=[j, j + 1])
                qml.RY(next(params), wires=j)
                qml.RY(next(params), wires=j + 1)
        return qml.probs(wires=0)
    return circuit


def _make_angle_circuit(n_qubits, depth, dev):
    @jax.jit
    @qml.qnode(dev, interface="jax")
    def circuit(network_params, x):
        params_per_layer = (n_qubits - 1) * 4
        theta_first = network_params[:n_qubits]
        theta_layers = network_params[n_qubits:n_qubits + depth * params_per_layer].reshape((depth, params_per_layer))
        alpha = network_params[n_qubits + depth * params_per_layer:]

        def encode_angles(scale):
            for w in range(n_qubits):
                qml.RY(scale * jnp.pi * x[w], wires=w)

        for w in range(n_qubits):
            qml.RY(theta_first[w], wires=w)
        encode_angles(alpha[0])

        @qml.for_loop(0, depth, 1)
        def layer_loop(i):
            layer_p = theta_layers[i]
            for j in range(n_qubits - 1):
                p = layer_p[j*4 : (j+1)*4]
                qml.CNOT(wires=[j, j + 1])
                qml.RY(p[0], wires=j)
                qml.RY(p[1], wires=j + 1)
                qml.CNOT(wires=[j, j + 1])
                qml.RY(p[2], wires=j)
                qml.RY(p[3], wires=j + 1)
            encode_angles(alpha[i + 1])

        layer_loop()
        return qml.probs(wires=0)
    return circuit


def _make_amplitude_circuit(n_qubits, depth, dev):
    @jax.jit
    @qml.qnode(dev, interface="jax")
    def circuit(network_params, x):
        params = iter(network_params)
        qml.AmplitudeEmbedding(features=x, wires=range(n_qubits), normalize=True, pad_with=0.0)
        for w in range(n_qubits):
            qml.RY(next(params), wires=w)
        for _ in range(depth):
            for j in range(n_qubits - 1):
                qml.CNOT(wires=[j, j + 1])
                qml.RY(next(params), wires=j)
                qml.RY(next(params), wires=j + 1)
                qml.CNOT(wires=[j, j + 1])
                qml.RY(next(params), wires=j)
                qml.RY(next(params), wires=j + 1)
        return qml.probs(wires=0)
    return circuit


_CIRCUIT_BUILDERS = {
    EncodingType.BASIS: _make_basis_circuit,
    EncodingType.ANGLE: _make_angle_circuit,
    EncodingType.AMPLITUDE: _make_amplitude_circuit,
}


def create_model(encoding: EncodingType, n_bits: int, depth: int):
    """Factory: returns (model, cfg) where model is a vmapped circuit."""
    n_qubits = get_n_qubits(encoding, n_bits)
    cfg = make_circuit_config(n_qubits, depth, encoding)
    dev = qml.device("default.qubit", wires=n_qubits)
    circuit = _CIRCUIT_BUILDERS[encoding](n_qubits, depth, dev)
    model = jax.vmap(circuit, in_axes=(None, 0))
    return model, cfg


SEED = args.seed
DEPTH = args.depth
ENCODING = EncodingType(args.encoding)

model, cfg = create_model(ENCODING, n_bits=args.n_bits, depth=DEPTH)
N_QUBITS = cfg.n_qubits
N_PARAMS_NETWORK = cfg.N_PARAMS_NETWORK

print("=" * 20)
print(f"Encoding: {ENCODING.value}")
print(f"n_bits={args.n_bits}, n_qubits={N_QUBITS}")
print(f"Total network parameters: {N_PARAMS_NETWORK}")
print("=" * 20)

key = jax.random.PRNGKey(SEED)


# Loss

# In[ ]:


def loss_acc(params, batch_x, batch_y):
    probs = model(params, batch_x)
    logits = jnp.log(probs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
    acc = (logits.argmax(-1) == batch_y).mean()
    return loss, acc


# # Training

# Sweep setup

# In[ ]:


EPOCHS = args.epochs
BATCH_SIZE = min(dataset.num_samples, 32)

load_dotenv()
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

print(f"WandB Entity: {WANDB_ENTITY}, Project: {WANDB_PROJECT}")
if not WANDB_ENTITY or not WANDB_PROJECT:
    raise ValueError(".env must define WANDB_ENTITY and WANDB_PROJECT for WandB logging.")

os.makedirs("plots", exist_ok=True)

enc_tag = ENCODING.value
sweep_config = {
    "name": f"parity_n{args.n_bits}_depth_{DEPTH}_epochs_{EPOCHS}_{enc_tag}",
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"value": 1e-2},
        "weight_decay": {"values": [float(v) for v in args.wd_coef]},
        "epochs": {"value": int(EPOCHS)},
        "batch_size": {"value": int(BATCH_SIZE)},
        "depth": {"value": int(DEPTH)},
        "n_qubits": {"value": int(N_QUBITS)},
        "n_bits": {"value": int(args.n_bits)},
        "seed": {"value": int(SEED)},
        "train_frac": {"values": [float(v) for v in args.train_frac]},
        "test_frac": {"value": float(args.test_frac)},
        "encoding": {"value": ENCODING.value},
    },
}

sweep_id = wandb.sweep(
    sweep=sweep_config,
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
)


# Training run

# In[ ]:


def run_training():
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
    cfg = run.config

    lr = float(cfg.learning_rate)
    wd_coef = float(cfg.weight_decay)
    run_epochs = int(cfg.epochs)
    train_frac = float(cfg.train_frac)
    test_frac = float(cfg.test_frac)

    split_data = dataset.get_random_split(training_frac=train_frac, test_frac=test_frac)
    train_X, train_Y = split_data["training"]
    val_X, val_Y = split_data["validation"]
    print(f"Train: {train_X.shape[0]}, Val: {val_X.shape[0]} (train_frac={train_frac})")

    opt = optax.adamw(learning_rate=lr, weight_decay=wd_coef)

    def make_batches(X, y, batch_size, batch_key):
        n = X.shape[0]
        n_full = n // batch_size
        perm = jax.random.permutation(batch_key, n)[: n_full * batch_size]
        Xb = X[perm].reshape(n_full, batch_size, *X.shape[1:])
        yb = y[perm].reshape(n_full, batch_size)
        return Xb, yb

    @jax.jit
    def train_epoch(params, opt_state, Xb, yb):
        def step(carry, batch):
            p, o = carry
            bx, by = batch
            (loss, acc), grads = jax.value_and_grad(loss_acc, has_aux=True)(p, bx, by)
            updates, o = opt.update(grads, o, p)
            p = optax.apply_updates(p, updates)
            return (p, o), (loss, acc)

        (params, opt_state), (losses, accs) = jax.lax.scan(step, (params, opt_state), (Xb, yb))
        return params, opt_state, jnp.mean(losses), jnp.mean(accs)

    @jax.jit
    def eval_epoch(params, Xb, yb):
        def step(_, batch):
            bx, by = batch
            loss, acc = loss_acc(params, bx, by)
            return None, (loss, acc)

        _, (losses, accs) = jax.lax.scan(step, None, (Xb, yb))
        return jnp.mean(losses), jnp.mean(accs)

    params = jnp.pi * jax.random.normal(key, (N_PARAMS_NETWORK,))
    opt_state = opt.init(params)

    train_loss_curve, val_loss_curve = [], []
    train_acc_curve, val_acc_curve = [], []

    run_key = key
    for epoch in range(1, run_epochs + 1):
        run_key, train_key, val_key = jax.random.split(run_key, 3)
        train_Xb, train_yb = make_batches(train_X, train_Y, BATCH_SIZE, train_key)
        val_Xb, val_yb = make_batches(val_X, val_Y, BATCH_SIZE, val_key)

        params, opt_state, tl, ta = train_epoch(params, opt_state, train_Xb, train_yb)
        vl, va = eval_epoch(params, val_Xb, val_yb)
        l2_norm = float(jnp.sqrt(jnp.sum(params**2)))

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
            }
        )

        if epoch % max(1, run_epochs // 100) == 0:
            print(
                f"Epoch {epoch:03d}/{run_epochs} | "
                f"train_loss={float(tl):.4f}, val_loss={float(vl):.4f}, "
                f"train_acc={float(ta):.4f}, val_acc={float(va):.4f}, "
                f"l2_norm={l2_norm:.4f}, wd={wd_coef}"
            )
        elif epoch == 1:
            print(f"Epoch {epoch}/{run_epochs} has completed.")
            print(
                f"Epoch {epoch:03d}/{run_epochs} | "
                f"train_loss={float(tl):.4f}, val_loss={float(vl):.4f}, "
                f"train_acc={float(ta):.4f}, val_acc={float(va):.4f}, "
                f"l2_norm={l2_norm:.4f}, wd={wd_coef}"
            )

        if epoch % max(1, run_epochs // 10) == 0:

                artifact = wandb.Artifact(
                    name=f"parity-model-n{args.n_bits}-depth-{DEPTH}-epochs-{run_epochs}-wd-{wd_coef}",
                    type="model",
                    metadata={"epoch": epoch, "val_loss": float(vl), "weight_decay": wd_coef},
                )

                flat_opt, _opt_tree_def = jax.tree.flatten(opt_state)
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

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
    ax[0].plot(train_loss_curve, label="Train")
    ax[0].plot(val_loss_curve, label="Validation")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(train_acc_curve, label="Train")
    ax[1].plot(val_acc_curve, label="Validation")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    plt.tight_layout()
    plot_path = f"plots/parity_n{args.n_bits}_depth_{DEPTH}_epochs_{run_epochs}_wd_{wd_coef}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {plot_path}")

    wandb.log({"training_curves": wandb.Image(plot_path)})
    wandb.finish()


# Run Sweep

# In[ ]:


print(f"Created W&B sweep: {sweep_id}")
wandb.agent(
    sweep_id=sweep_id,
    function=run_training,
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    count=args.sweep_count,
)

