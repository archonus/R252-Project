"""
Low-depth quantum circuit MNIST classifier.

A training script following the parity_grokking style, with W&B logging,
argparse, and state caching.
"""

import os
import argparse

import numpy
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
from pennylane import numpy as pnp
from dotenv import load_dotenv
import wandb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# FRQI / MCRQI encoding and decoding
# ---------------------------------------------------------------------------

def FRQI_encoding(images):
    """
    Input : (batchsize, N, N) ndarray — grayscale images.
    Returns : (batchsize, 2, N**2) ndarray — FRQI quantum states.
    """
    batchsize, N, _ = images.shape
    n = 2 * int(pnp.log2(N))
    states = pnp.reshape(images, (batchsize, *(2,) * n))
    states = pnp.transpose(
        states, [0] + [ax + 1 for q in range(n // 2) for ax in (q, q + n // 2)]
    )
    states = pnp.stack(
        [pnp.cos(pnp.pi / 2 * states), pnp.sin(pnp.pi / 2 * states)], axis=1
    )
    states = pnp.reshape(states, (batchsize, 2, N**2)) / N
    return states


def FRQI_decoding(states):
    """
    Input : (batchsize, 2, N**2) ndarray — FRQI quantum states.
    Returns : (batchsize, N, N) ndarray — grayscale images.
    """
    batchsize = states.shape[0]
    states = pnp.reshape(states, (batchsize, 2, -1))
    n = int(pnp.log2(states.shape[2]))
    images = pnp.arccos(
        states[:, 0] ** 2 * 2**n - states[:, 1] ** 2 * 2**n
    ) / pnp.pi
    images = pnp.reshape(images, (batchsize, *(2,) * n))
    images = pnp.transpose(images, [0, *range(1, n, 2), *range(2, n + 1, 2)])
    images = pnp.reshape(images, (batchsize, 2 ** (n // 2), 2 ** (n // 2)))
    return images


# def MCRQI_encoding(images):
#     """
#     Input : (batchsize, N, N, 3) ndarray — RGB images.
#     Returns : (batchsize, 8, N**2) ndarray — MCRQI quantum states.
#     """
#     batchsize, N, _, channels = images.shape
#     n = 2 * int(pnp.log2(N))
#     states = pnp.reshape(images, (batchsize, *(2,) * n, channels))
#     states = pnp.transpose(
#         states,
#         [0] + [ax + 1 for q in range(n // 2) for ax in (q, q + n // 2)] + [n + 1],
#     )
#     states = pnp.stack(
#         [
#             pnp.cos(pnp.pi / 2 * states[..., 0]),
#             pnp.cos(pnp.pi / 2 * states[..., 1]),
#             pnp.cos(pnp.pi / 2 * states[..., 2]),
#             pnp.ones(states.shape[:-1]),
#             pnp.sin(pnp.pi / 2 * states[..., 0]),
#             pnp.sin(pnp.pi / 2 * states[..., 1]),
#             pnp.sin(pnp.pi / 2 * states[..., 2]),
#             pnp.zeros(states.shape[:-1]),
#         ],
#         axis=1,
#     )
#     states = pnp.reshape(states, (batchsize, 8, N**2)) / (2 * N)
#     return states


# def MCRQI_decoding(states):
#     """
#     Input : (batchsize, 8, N**2) ndarray — MCRQI quantum states.
#     Returns : (batchsize, N, N, 3) ndarray — RGB images.
#     """
#     batchsize = states.shape[0]
#     states = pnp.reshape(states, (batchsize, 8, -1))
#     N2 = states.shape[2]
#     N = int(pnp.sqrt(N2))
#     n = int(pnp.log2(N2))
#     images = pnp.arccos(
#         states[:, :3] ** 2 * 4 * N2 - states[:, 4:7] ** 2 * 4 * N2
#     ) / pnp.pi
#     images = pnp.reshape(images, (batchsize, 3, *(2,) * n))
#     images = pnp.transpose(
#         images, [0, *range(2, n + 1, 2), *range(3, n + 2, 2), 1]
#     )
#     images = pnp.reshape(images, (batchsize, N, N, 3))
#     return images


# ---------------------------------------------------------------------------
# State preparation circuit (from dataset params)
# ---------------------------------------------------------------------------

def get_state_prep_circuit(circuit_layout, n_qubits):
    """Build a JIT-compiled circuit that maps dataset params -> quantum state."""
    jax.config.update("jax_enable_x64", True)  # float64 for state prep accuracy
    dev = qml.device("default.qubit", wires=n_qubits)

    @jax.jit
    @qml.qnode(dev)
    def circuit(params):
        counter = 0
        for gate, wire in circuit_layout:
            if gate == "RY":
                qml.RY(params[counter], wire)
                counter += 1
            elif gate == "CNOT":
                qml.CNOT(wire)
        return qml.state()

    return circuit


def compute_or_load_states(args, dataset_params, selection, circuit_layout):
    """
    Compute quantum states from circuit parameters, caching to disk.
    Uses float64 for state preparation, then casts to float32 for training.
    """
    labels_tag = "_".join(str(l) for l in sorted(args.target_labels))
    cache_path = os.path.join(
        args.cache_dir,
        f"{args.dataset_name}_d{args.circuit_depth}_labels_{labels_tag}.npz",
    )

    if os.path.exists(cache_path):
        print(f"Loading cached states from {cache_path}")
        data = numpy.load(cache_path)
        states = jnp.asarray(data["states"], dtype=jnp.float32)
        labels = jnp.asarray(data["labels"], dtype=jnp.int32)
        fidelities = data["fidelities"]
        exact_state = data["exact_state"]
        return states, labels, fidelities, exact_state

    # Compute from scratch
    params_key = f"params_d{args.circuit_depth}"
    fidelities_key = f"fidelities_d{args.circuit_depth}"
    all_params = pnp.asarray(getattr(dataset_params, params_key))[selection]

    n_qubits = len(set(w if isinstance(w, int) else w[0] for _, w in circuit_layout))
    circuit = get_state_prep_circuit(circuit_layout, n_qubits)

    n = len(all_params)
    states_list = []
    print("Computing quantum states from circuit parameters...")
    for i, p in enumerate(all_params):
        states_list.append(circuit(p))
        if (i + 1) % max(1, n // 10) == 0:
            print(f"  {(i + 1) / n * 100:.0f}% computed")

    # Gather results
    states_f64 = pnp.asarray(states_list)
    labels_all = pnp.asarray(dataset_params.labels)[selection]
    fidelities = pnp.asarray(getattr(dataset_params, fidelities_key))[selection]
    exact_state = pnp.asarray(dataset_params.exact_state)[selection]

    # Cache to disk
    os.makedirs(args.cache_dir, exist_ok=True)
    numpy.savez(
        cache_path,
        states=numpy.asarray(states_f64.real),
        labels=numpy.asarray(labels_all),
        fidelities=numpy.asarray(fidelities),
        exact_state=numpy.asarray(exact_state),
    )
    print(f"Cached states to {cache_path}")

    # Cast to float32 for training
    states = jnp.asarray(states_f64.real, dtype=jnp.float32)
    labels = jnp.asarray(labels_all, dtype=jnp.int32)
    return states, labels, fidelities, exact_state


# ---------------------------------------------------------------------------
# VQC classifier
# ---------------------------------------------------------------------------

def build_classifier(n_qubits, depth, n_params):
    """Build the variational quantum classifier circuit."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @jax.jit
    @qml.qnode(dev, interface="jax")
    def classifier(network_params, state):
        p = iter(network_params)
        qml.StatePrep(state, wires=range(n_qubits))

        for w in range(n_qubits):
            qml.RY(next(p), wires=w)

        for _ in range(depth):
            for j in range(n_qubits - 1):
                qml.CNOT(wires=[j, j + 1])
                qml.RY(next(p), wires=j)
                qml.RY(next(p), wires=j + 1)
                qml.CNOT(wires=[j, j + 1])
                qml.RY(next(p), wires=j)
                qml.RY(next(p), wires=j + 1)

        return qml.probs(n_qubits - 1)

    return classifier


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_batches(X, y, batch_size, batch_key):
    n = X.shape[0]
    n_full = n // batch_size
    perm = jax.random.permutation(batch_key, n)[:n_full * batch_size]
    Xb = X[perm].reshape(n_full, batch_size, *X.shape[1:])
    yb = y[perm].reshape(n_full, batch_size)
    return Xb, yb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    defaults = {
        "dataset_name": "low-depth-mnist",
        "target_labels": [0, 1],
        "circuit_depth": 4,
        "classifier_depth": 4,
        "epochs": 5,
        "batch_size": 128,
        "val_frac": [0.2],
        "wd_coef": [1],
        "lr": [1e-2],
        "seed": 0,
        "cache_dir": "datasets",
        "sweep_count": None,
    }

    parser = argparse.ArgumentParser(
        description="Low-depth quantum circuit MNIST classifier."
    )
    parser.add_argument(
        "--dataset_name", type=str, default=defaults["dataset_name"],
        help="PennyLane dataset name",
    )
    parser.add_argument(
        "--target_labels", type=int, nargs="+", default=defaults["target_labels"],
        help="Digit labels to classify (e.g. 0 1 or 3 5 7)",
    )
    parser.add_argument(
        "--circuit_depth", type=int, default=defaults["circuit_depth"],
        help="State-preparation circuit depth (4 or 8)",
    )
    parser.add_argument(
        "--classifier_depth", type=int, default=defaults["classifier_depth"],
        help="Number of SO(4) block layers in the VQC classifier",
    )
    parser.add_argument(
        "--epochs", type=int, default=defaults["epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=defaults["batch_size"],
        help="Mini-batch size",
    )
    parser.add_argument(
        "--val_frac", type=float, nargs="+", default=defaults["val_frac"],
        help="Validation fraction(s). Multiple values trigger a W&B sweep.",
    )
    parser.add_argument(
        "--wd_coef", type=float, nargs="+", default=defaults["wd_coef"],
        help="Weight decay coefficient(s). Multiple values trigger a W&B sweep.",
    )
    parser.add_argument(
        "--lr", type=float, nargs="+", default=defaults["lr"],
        help="Learning rate(s). Multiple values trigger a W&B sweep.",
    )
    parser.add_argument(
        "--seed", type=int, default=defaults["seed"],
        help="Random seed",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=defaults["cache_dir"],
        help="Directory for caching pre-computed quantum states",
    )
    parser.add_argument(
        "--sweep_count", type=int, default=defaults["sweep_count"],
        help="Max number of sweep runs (None = run all)",
    )
    return parser.parse_args()


def needs_sweep(args):
    """Return True if any hyperparameter has multiple values to search."""
    return len(args.wd_coef) > 1 or len(args.lr) > 1 or len(args.val_frac) > 1


def main():
    args = parse_args()
    jax.config.update("jax_platform_name", "cpu")

    # ---- Load / download dataset ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.cache_dir = os.path.join(script_dir, args.cache_dir)
    dataset_path = os.path.join(script_dir, "datasets", args.dataset_name, f"{args.dataset_name}.h5")
    if os.path.exists(dataset_path):
        dataset_params = qml.data.Dataset.open(dataset_path)
    else:
        print(f"Downloading dataset {args.dataset_name} (~1 GB)...")
        [dataset_params] = qml.data.load(args.dataset_name)

    # ---- Select target labels ----
    labels = pnp.asarray(dataset_params.labels)
    selection = pnp.isin(labels, args.target_labels)

    layout_key = f"circuit_layout_d{args.circuit_depth}"
    circuit_layout = getattr(dataset_params, layout_key)

    # ---- Compute or load cached states ----
    states, labels_sel, fidelities, exact_state = compute_or_load_states(
        args, dataset_params, selection, circuit_layout
    )

    # After state prep, disable float64 for faster training
    jax.config.update("jax_enable_x64", False)

    n_classes = len(args.target_labels)
    n_qubits = 11  # fixed by the dataset (32x32 grayscale FRQI) int(jnp.log2(states.shape[-1]))

    # Remap labels to 0..n_classes-1
    sorted_labels = jnp.array(sorted(args.target_labels))
    label_map = jnp.zeros(int(sorted_labels.max()) + 1, dtype=jnp.int32)
    for new_idx, old_label in enumerate(sorted_labels):
        label_map = label_map.at[old_label].set(new_idx)
    labels_sel = label_map[labels_sel]

    # ---- Build classifier ----
    n_params_first_layer = n_qubits
    n_params_block = 4
    n_params_network = n_params_first_layer + (n_qubits - 1) * args.classifier_depth * n_params_block

    classifier = build_classifier(n_qubits, args.classifier_depth, n_params_network)
    model = jax.vmap(classifier, in_axes=(None, 0))

    def loss_acc(params, batch_x, batch_y):
        logits = model(params, batch_x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
        acc = (logits.argmax(-1) == batch_y).mean()
        return loss, acc

    # ---- W&B setup ----
    load_dotenv()
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    if not WANDB_ENTITY or not WANDB_PROJECT:
        raise ValueError(".env must define WANDB_ENTITY and WANDB_PROJECT")

    os.makedirs("plots", exist_ok=True)

    labels_tag = "_".join(str(l) for l in sorted(args.target_labels))

    # ---- Shared training function ----
    def run_training(config=None):
        """Single training run. Called directly or via wandb.agent."""
        if config is not None:
            # Called as a sweep agent
            run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
            cfg = run.config
        else:
            # Direct single run — cfg comes from args
            cfg = argparse.Namespace(
                learning_rate=args.lr[0],
                weight_decay=args.wd_coef[0],
                val_frac=args.val_frac[0],
            )
            run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=f"mnist_{labels_tag}_d{args.circuit_depth}_cd{args.classifier_depth}_wd{cfg.weight_decay}_vf{cfg.val_frac}",
                config={
                    "learning_rate": cfg.learning_rate,
                    "weight_decay": cfg.weight_decay,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "classifier_depth": args.classifier_depth,
                    "circuit_depth": args.circuit_depth,
                    "n_qubits": n_qubits,
                    "target_labels": args.target_labels,
                    "seed": args.seed,
                    "val_frac": cfg.val_frac,
                },
            )

        lr = float(cfg.learning_rate)
        wd_coef = float(cfg.weight_decay)
        val_frac = float(cfg.val_frac)
        run_epochs = args.epochs
        batch_size = args.batch_size

        # ---- Train / val split (per-run, so val_frac can be swept) ----
        key = jax.random.PRNGKey(args.seed)
        key, key_perm = jax.random.split(key)
        perm = jax.random.permutation(key_perm, len(states))
        split_pt = int(len(states) * (1 - val_frac))
        train_idx, val_idx = perm[:split_pt], perm[split_pt:]

        X_train, y_train = states[train_idx], labels_sel[train_idx]
        X_val, y_val = states[val_idx], labels_sel[val_idx]
        print(f"Data: {len(X_train)} train, {len(X_val)} val, {n_classes} classes")

        opt = optax.adamw(learning_rate=lr, weight_decay=wd_coef)

        @jax.jit
        def train_epoch(params, opt_state, Xb, yb):
            def step(carry, batch):
                p, o = carry
                bx, by = batch
                (loss, acc), grads = jax.value_and_grad(loss_acc, has_aux=True)(p, bx, by)
                updates, o = opt.update(grads, o, p)
                p = optax.apply_updates(p, updates)
                return (p, o), (loss, acc)

            (params, opt_state), (losses, accs) = jax.lax.scan(
                step, (params, opt_state), (Xb, yb)
            )
            return params, opt_state, jnp.mean(losses), jnp.mean(accs)

        @jax.jit
        def eval_epoch(params, Xb, yb):
            def step(_, batch):
                bx, by = batch
                loss, acc = loss_acc(params, bx, by)
                return None, (loss, acc)

            _, (losses, accs) = jax.lax.scan(step, None, (Xb, yb))
            return jnp.mean(losses), jnp.mean(accs)

        # Init params
        run_key = jax.random.PRNGKey(args.seed)
        params = 2 * jnp.pi * jax.random.uniform(
            run_key, (n_params_network,), dtype=jnp.float32
        )
        opt_state = opt.init(params)

        train_loss_curve, val_loss_curve = [], []
        train_acc_curve, val_acc_curve = [], []

        rng = run_key
        for epoch in range(1, run_epochs + 1):
            rng, train_key, val_key = jax.random.split(rng, 3)
            train_Xb, train_yb = make_batches(X_train, y_train, batch_size, train_key)
            val_Xb, val_yb = make_batches(X_val, y_val, batch_size, val_key)

            params, opt_state, tl, ta = train_epoch(params, opt_state, train_Xb, train_yb)
            vl, va = eval_epoch(params, val_Xb, val_yb)
            l2_norm = float(jnp.sqrt(jnp.sum(params**2)))

            train_loss_curve.append(tl)
            val_loss_curve.append(vl)
            train_acc_curve.append(ta)
            val_acc_curve.append(va)

            wandb.log({
                "epoch": epoch,
                "train_loss": float(tl),
                "val_loss": float(vl),
                "train_acc": float(ta),
                "val_acc": float(va),
                "param_l2_norm": l2_norm,
            })

            if epoch % max(1, run_epochs // 100) == 0:
                print(
                    f"Epoch {epoch:03d}/{run_epochs} | "
                    f"train_loss={float(tl):.4f}, val_loss={float(vl):.4f}, "
                    f"train_acc={float(ta):.4f}, val_acc={float(va):.4f}, "
                    f"l2_norm={l2_norm:.4f}, wd={wd_coef}"
                )

                # Checkpoint
                artifact = wandb.Artifact(
                    name=f"mnist-model-{labels_tag}-cd{args.classifier_depth}-d{args.circuit_depth}-wd{wd_coef}",
                    type="model",
                    metadata={"epoch": epoch, "val_loss": float(vl), "weight_decay": wd_coef},
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

            elif epoch == 1:
                print(
                    f"Epoch {epoch:03d}/{run_epochs} | "
                    f"train_loss={float(tl):.4f}, val_loss={float(vl):.4f}, "
                    f"train_acc={float(ta):.4f}, val_acc={float(va):.4f}, "
                    f"l2_norm={l2_norm:.4f}, wd={wd_coef}"
                )

        # ---- Save plots ----
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
        plot_path = f"plots/mnist_{labels_tag}_cd{args.classifier_depth}_d{args.circuit_depth}_epochs{run_epochs}_wd{wd_coef}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plots saved to {plot_path}")

        wandb.log({"training_curves": wandb.Image(plot_path)})
        wandb.finish()

    # ---- Launch: single run or sweep ----
    if needs_sweep(args):
        sweep_config = {
            "name": f"mnist_{labels_tag}_cd{args.classifier_depth}_d{args.circuit_depth}_epochs{args.epochs}",
            "method": "grid",
            "metric": {"name": "val_loss", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"values": [float(v) for v in args.lr]},
                "weight_decay": {"values": [float(v) for v in args.wd_coef]},
                "val_frac": {"values": [float(v) for v in args.val_frac]},
            },
        }
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
        )
        print(f"Created W&B sweep: {sweep_id}")
        wandb.agent(
            sweep_id=sweep_id,
            function=lambda: run_training(config=True),
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            count=args.sweep_count,
        )
    else:
        print("Single hyperparameter config — running directly (no sweep).")
        run_training()


if __name__ == "__main__":
    main()
