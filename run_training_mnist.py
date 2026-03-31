from dotenv import load_dotenv
import os
import wandb
import argparse


from quantum_transformers.utils import plot_image
from quantum_transformers.datasets import get_mnist_dataloaders
from quantum_transformers.training import train_and_evaluate
from quantum_transformers.transformers import VisionTransformer
from quantum_transformers.quantum_layer import get_circuit

load_dotenv()
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

def run_training():
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
    cfg = run.config
    data_dir = './data'
    mnist_train_dataloader, mnist_valid_dataloader, mnist_test_dataloader = get_mnist_dataloaders(
        data_dir, 
        batch_size=cfg.batch_size,
        train_frac=cfg.train_frac, 
    )

    model = VisionTransformer(
        num_classes=10, 
        patch_size=14, 
        hidden_size=6, 
        num_heads=2, 
        num_transformer_blocks=4, 
        mlp_hidden_size=3, 
        quantum_attn_circuit=get_circuit(), 
        quantum_mlp_circuit=get_circuit()
    )

    train_and_evaluate(
        model, 
        mnist_train_dataloader, 
        mnist_valid_dataloader, 
        mnist_test_dataloader, 
        num_classes=10, 
        num_epochs=cfg.epochs,
        weight_decay=cfg.wd_coef,
        seed=cfg.seed,
        wandb_run=run,
    )

    wandb.finish()



defaults = {
    "epochs": 30,
    "wd_coef": [1e-4, 1e-5],
    "sweep_count": 10,
    "train_frac": [0.2, 0.4, 0.8],
    "seed": 42,
}

parser = argparse.ArgumentParser(description="Grokking with quantum vision transformer.")
parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="Number of training epochs")
parser.add_argument("--wd_coef", type=float, nargs="+", default=defaults["wd_coef"], help="Weight decay values for W&B sweep")
parser.add_argument("--sweep_count", type=int, default=defaults["sweep_count"], help="Max number of sweep runs")
parser.add_argument("--train_frac", type=float, nargs="+", default=defaults["train_frac"], help="Fraction of training data to use in W&B sweep. Note that the MNIST test dataset is a separate dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed")

if __name__ == "__main__":

    args = parser.parse_args()

    print(f"WandB Entity: {WANDB_ENTITY}, Project: {WANDB_PROJECT}")

    if not WANDB_ENTITY or not WANDB_PROJECT:
        raise ValueError(".env must define WANDB_ENTITY and WANDB_PROJECT for WandB logging.")
    
    sweep_config = {
        'name': f'quantum_transformer_mnist',
        'method': 'grid',
        "metric": {"name": "val_loss", "goal": "minimize"},
        'parameters': {
            'wd_coef': {'values': args.wd_coef},
            'train_frac': {'values': args.train_frac},
            'epochs': {'value': args.epochs},
            'batch_size': {'value': args.batch_size},
            'seed': {'value': args.seed},
            'train_frac': {'values': [float(x) for x in args.train_frac]}
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )

    print(f"Created W&B sweep: {sweep_id}")
    wandb.agent(sweep_id, function=run_training, count=args.sweep_count, entity=WANDB_ENTITY, project=WANDB_PROJECT)



    





