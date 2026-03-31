from dotenv import load_dotenv
import os
import wandb
import argparse
import operator

import jax.numpy as jnp


from cayley_table import CayleyTableDataset
from quantum_transformers.training import train_and_evaluate
# from quantum_transformers.transformers import Transformer
from quantum_transformers.dressed_transformers import Transformer
from quantum_transformers.quantum_layer import get_circuit
from quantum_transformers.utils import pretty_print_dict

load_dotenv()
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")


def _to_batches(x, y, batch_size: int):
    batches = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        if len(xb) == 0:
            continue
        batches.append((jnp.array(xb, dtype=jnp.int32), jnp.array(yb, dtype=jnp.int32)))
    return batches

def run_training():
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
    cfg = run.config
    op_map = {
        "add": operator.add,
        "mul": operator.mul,
        "sub": operator.sub,
    }
    operation_name = str(cfg.operation)
    if operation_name not in op_map:
        raise ValueError(f"Unsupported operation '{operation_name}'. Choose from {list(op_map)}")
    
    print("Running training with config:")
    pretty_print_dict(cfg)

    dataset = CayleyTableDataset(p=int(cfg.prime), seed=int(cfg.seed), op=op_map[operation_name])
    splits = dataset.get_random_split(training_frac=float(cfg.train_frac), test_frac=float(cfg.test_frac))

    train_x, train_y = splits["training"]
    if "validation" in splits:
        val_x, val_y = splits["validation"]
    else:
        val_x, val_y = splits["test"]
    test_x, test_y = splits["test"]

    train_dataloader = _to_batches(train_x, train_y, int(cfg.batch_size))
    val_dataloader = _to_batches(val_x, val_y, int(cfg.batch_size))
    test_dataloader = _to_batches(test_x, test_y, int(cfg.batch_size))

    # Determine quantum circuits based on quantum_mode
    quantum_attn_circuit = None
    quantum_mlp_circuit = None
    if cfg.quantum_mode in ["attention", "both"]:
        quantum_attn_circuit = get_circuit()
    if cfg.quantum_mode in ["mlp", "both"]:
        quantum_mlp_circuit = get_circuit()

    model = Transformer(
        num_tokens=int(cfg.prime),
        max_seq_len=2,
        num_classes=int(cfg.prime),
        hidden_size=int(cfg.hidden_size),
        num_heads=int(cfg.num_heads),
        num_transformer_blocks=int(cfg.num_transformer_blocks),
        mlp_hidden_size=int(cfg.mlp_hidden_size),
        quantum_attn_circuit=quantum_attn_circuit,
        quantum_mlp_circuit=quantum_mlp_circuit,
        quantum_num_qubits=int(cfg.num_qubits)
    )

    train_and_evaluate(
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader=val_dataloader if len(test_dataloader) == 0 else test_dataloader,  # Use validation set as test set if no separate test set is provided
        num_classes=int(cfg.prime),
        num_epochs=cfg.epochs,
        weight_decay=cfg.wd_coef,
        seed=cfg.seed,
        wandb_run=run,
        checkpoint_directory=f"./checkpoints/cayley_{operation_name}_p{cfg.prime}_train{cfg.train_frac}_test{cfg.test_frac}_wd{cfg.wd_coef}_run{run.id}",
    )

defaults = {
    "epochs": 1000,
    "wd_coef": [1, 0.5],
    "sweep_count": None,
    "train_frac": [0.8, 0.5, 0.3],
    "test_frac": 0.0,
    "seed": 42,
    "prime": 13,
    "batch_size": 64,
    "hidden_size": [128],
    "num_heads": [4],
    "num_transformer_blocks": [2],
    "mlp_hidden_size": [512],
    "operation": "add",
    "quantum_mode": "both",
    "num_qubits": 4,
}

parser = argparse.ArgumentParser(description="Grokking with transformer on algorithmic datasets.")
parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="Number of training epochs")
parser.add_argument("--wd_coef", type=float, nargs="+", default=defaults["wd_coef"], help="Weight decay values for W&B sweep")
parser.add_argument("--sweep_count", type=int, default=defaults["sweep_count"], help="Max number of sweep runs")
parser.add_argument("--train_frac", type=float, nargs="+", default=defaults["train_frac"], help="Fraction of Cayley table entries used for training")
parser.add_argument("--test_frac", type=float, default=defaults["test_frac"], help="Fraction of Cayley table entries used for testing")
parser.add_argument("--batch_size", type=int, default=defaults["batch_size"], help="Batch size for training")
parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed")
parser.add_argument("--prime", type=int, default=defaults["prime"], help="Prime number for modular arithmetic in the dataset")
parser.add_argument("--operation", type=str, default=defaults["operation"], choices=["add", "mul", "sub"], help="Operation for the dataset: add, mul, or sub")
parser.add_argument("--hidden_size", type=int, nargs="+", default=defaults["hidden_size"], help="Hidden size for the transformer")
parser.add_argument("--num_heads", type=int, nargs="+", default=defaults["num_heads"], help="Number of attention heads in the transformer")
parser.add_argument("--num_transformer_blocks", type=int, nargs="+", default=defaults["num_transformer_blocks"], help="Number of transformer blocks")
parser.add_argument("--mlp_hidden_size", type=int, nargs="+", default=defaults["mlp_hidden_size"], help="Hidden size for the MLP in the transformer")
parser.add_argument("--quantum_mode", type=str, default=defaults["quantum_mode"], choices=["none", "attention", "mlp", "both"], help="Quantum layer configuration: none, attention only, mlp only, or both (default)")
parser.add_argument("--num_qubits", type=int, default=defaults["num_qubits"], help="Number of qubits for quantum layers (if used)")



if __name__ == "__main__":

    args = parser.parse_args()

    print(f"WandB Entity: {WANDB_ENTITY}, Project: {WANDB_PROJECT}")
    print(f"Chosen quantum mode: {args.quantum_mode}")
    if not WANDB_ENTITY or not WANDB_PROJECT:
        raise ValueError(".env must define WANDB_ENTITY and WANDB_PROJECT for WandB logging.")
    suffix = f"cayley_p{args.prime}_epochs{args.epochs}"
    sweep_config = {
        'name': f"{args.quantum_mode}_dressed_transformer_" + suffix,
        'method': 'grid',
        "metric": {"name": "val_loss", "goal": "minimize"},
        'parameters': {
            'wd_coef': {'values': [float(x) for x in args.wd_coef]},
            'train_frac': {'values': [float(x) for x in args.train_frac]},
            'test_frac': {'value': float(args.test_frac)},
            'epochs': {'value': args.epochs},
            'batch_size': {'value': args.batch_size},
            'seed': {'value': args.seed},
            'prime': {'value': args.prime},
            'operation': {'value': args.operation},
            'quantum_mode': {'value': args.quantum_mode},
            'hidden_size': {'values': [int(x) for x in args.hidden_size]},
            'num_heads': {'values': [int(x) for x in args.num_heads]},
            'num_transformer_blocks': {'values': [int(x) for x in args.num_transformer_blocks]},
            'mlp_hidden_size': {'values': [int(x) for x in args.mlp_hidden_size]},
            'num_qubits': {'value': int(args.num_qubits)}
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )

    print(f"Created W&B sweep: {sweep_id}")
    wandb.agent(sweep_id, function=run_training, count=args.sweep_count, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    wandb.finish()



    





