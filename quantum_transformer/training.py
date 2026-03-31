from typing import Optional, Any
import time

import numpy.typing as npt
import jax
import jax.numpy as jnp
import flax.linen
import flax.training.train_state
import flax.core
import optax
from tqdm import tqdm
import orbax.checkpoint as ocp
import os

import wandb

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'


class TrainState(flax.training.train_state.TrainState):
    # See https://flax.readthedocs.io/en/latest/guides/dropout.html.
    key: jax.random.KeyArray  # type: ignore


@jax.jit
def train_step(state: TrainState, inputs: jax.Array, labels: jax.Array, key: jax.random.KeyArray) -> TrainState:
    """
    Performs a single training step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs: The batch of inputs.
        labels: The batch of labels.
        key: The random key to use.

    Returns:
        The updated training state.
    """
    key, dropout_key = jax.random.split(key=key)
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            x=inputs,
            train=True,
            rngs={'dropout': dropout_train_key}
        )
        if logits.shape[1] <= 2:
            if logits.shape[1] == 2:
                logits = logits[:, 1]
            loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels).mean()
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
        # return loss, logits
        return loss
    # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # (loss, logits), grads = grad_fn(state.params)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def eval_step(state: TrainState, inputs: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Performs a single evaluation step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs: The batch of inputs.
        labels: The batch of labels.

    Returns:
        loss: The loss on the given batch.
        logits: The logits on the given batch.
    """
    logits = state.apply_fn(
        {'params': state.params},
        x=inputs,
        train=False,
        rngs={'dropout': state.key}
    )
    if logits.shape[1] <= 2:
        if logits.shape[1] == 2:
            logits = logits[:, 1]
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels).mean()
    else:
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    return loss, logits


def evaluate(state: TrainState, eval_dataloader, num_classes: int,
             tqdm_desc: Optional[str] = None, debug: bool = False) -> tuple[float, float, npt.ArrayLike, npt.ArrayLike]:
    """
    Evaluates the model given the current training state on the given dataloader.

    Args:
        state: The current training state.
        eval_dataloader: The dataloader to evaluate on.
        num_classes: The number of classes.
        tqdm_desc: The description to use for the tqdm progress bar. If None, no progress bar is shown.
        debug: Whether to print extra information for debugging.

    Returns:
        eval_loss: The loss.
        eval_accuracy: The accuracy.
    """
    logits, labels = [], []
    eval_loss = 0.0
    with tqdm(total=len(eval_dataloader), desc=tqdm_desc, unit="batch", bar_format=TQDM_BAR_FORMAT, disable=tqdm_desc is None) as progress_bar:
        for inputs_batch, labels_batch in eval_dataloader:
            loss_batch, logits_batch = eval_step(state, inputs_batch, labels_batch)
            logits.append(logits_batch)
            labels.append(labels_batch)
            eval_loss += loss_batch
            progress_bar.update(1)
        eval_loss /= len(eval_dataloader)
        logits = jnp.concatenate(logits)  # type: ignore
        y_true = jnp.concatenate(labels)  # type: ignore
        if debug:
            print(f"logits = {logits}")
        predictions = jnp.argmax(logits, axis=-1)
        eval_accuracy = float(jnp.mean(predictions == y_true))
        if debug:
            print(f"predictions = {predictions}")
            print(f"y_true = {y_true}")
        eval_fpr, eval_tpr = [], []
        progress_bar.set_postfix_str(f"Loss = {eval_loss:.4f}, Accuracy = {eval_accuracy:.3f}")
    return float(eval_loss), eval_accuracy, eval_fpr, eval_tpr

def scale_omnigrok_ffn(params, alpha):
    """
    Scales the weights and biases of FeedForward layers by a factor of alpha.
    """
    def scale_fn(path, value):
        # Extract the string keys from the path tuple
        path_keys = [p.key for p in path if hasattr(p, 'key')]
        
        # Check if any key in the current path indicates a FeedForward layer
        if any(isinstance(k, str) and k.startswith('FeedForward') for k in path_keys):
            return value * alpha
        return value

    # Map the scaling function over the entire parameter tree
    scaled_params = jax.tree_util.tree_map_with_path(scale_fn, params)
    return scaled_params

def train_and_evaluate(
        model: flax.linen.Module, 
        train_dataloader, 
        val_dataloader, 
        test_dataloader, 
        num_classes: int,
        num_epochs: int,
        learning_rate_or_schedule = None,
        weight_decay: float = 1e-4,
        init_scale: float = 1.0,
        seed: int = 42, 
        wandb_run = None,
        save_interval: int = 5,
        checkpoint_directory = "./local_checkpoints",
        debug: bool = False
    ) -> dict[str, Any]:
    """
    Trains the given model on the given dataloaders for the given hyperparameters.

    The progress and evaluation results are printed to stdout.

    Args:
        model: The model to train.
        train_dataloader: The dataloader for the training set.
        val_dataloader: The dataloader for the validation set.
        num_classes: The number of classes.I'm 
        num_epochs: The number of epochs to train for.
        seed: The seed to use for reproducibility.
        wandb_run: Run object for logging to Weights & Biases. If None, no logging is done.
        debug: Whether to print extra information for debugging.
    """
    checkpoint_options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(checkpoint_directory),
        checkpointers=ocp.PyTreeCheckpointer(),
        options=checkpoint_options
    )
    root_key = jax.random.PRNGKey(seed=seed)
    root_key, params_key, train_key = jax.random.split(key=root_key, num=3)

    dummy_batch = next(iter(train_dataloader))[0]
    input_shape = dummy_batch[0].shape
    input_dtype = dummy_batch[0].dtype
    batch_size = len(dummy_batch)
    root_key, input_key = jax.random.split(key=root_key)
    if jnp.issubdtype(input_dtype, jnp.floating):
        dummy_batch = jax.random.uniform(key=input_key, shape=(batch_size,) + tuple(input_shape), dtype=input_dtype)
    elif jnp.issubdtype(input_dtype, jnp.integer):
        dummy_batch = jax.random.randint(key=input_key, shape=(batch_size,) + tuple(input_shape), minval=0, maxval=100, dtype=input_dtype)
    else:
        raise ValueError(f"Unsupported dtype {input_dtype}")

    variables = model.init(params_key, dummy_batch, train=False)

    variables['params'] = scale_omnigrok_ffn(variables['params'], init_scale)

    if debug:
        print(jax.tree_map(lambda x: x.shape, variables))
    print(f"Number of parameters = {sum(x.size for x in jax.tree_util.tree_leaves(variables))}")

    if learning_rate_or_schedule is None:
        learning_rate_or_schedule = create_learning_rate_schedule()

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate_or_schedule, weight_decay=weight_decay),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        key=train_key,
        tx=optimizer
    )

    best_val_accuracy, best_epoch, best_state = 0.0, 0, None
    total_train_time = 0.0
    start_time = time.time()

    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'train_val_loss_gap': [],
        'train_val_acc_gap': [],
        'test_loss': 0.0,
        'test_accuracy': 0.0,
        'test_fpr': [],
        'test_tpr': [],
    }

    for epoch in range(num_epochs):
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1:3}/{num_epochs}", unit="batch", bar_format=TQDM_BAR_FORMAT) as progress_bar:
            epoch_train_time = time.time()
            for inputs_batch, labels_batch in train_dataloader:
                state = train_step(state, inputs_batch, labels_batch, train_key)
                progress_bar.update(1)
            epoch_train_time = time.time() - epoch_train_time
            total_train_time += epoch_train_time

            train_loss, train_accuracy, _, _ = evaluate(state, train_dataloader, num_classes, tqdm_desc=None, debug=debug)
            val_loss, val_accuracy, _, _ = evaluate(state, val_dataloader, num_classes, tqdm_desc=None, debug=debug)
            progress_bar.set_postfix_str(f"Loss = {val_loss:.4f}, Accuracy = {val_accuracy:.3f}, Train time = {epoch_train_time:.2f}s")

            metrics['train_losses'].append(train_loss)
            metrics['val_losses'].append(val_loss)
            metrics['train_accuracies'].append(train_accuracy)
            metrics['val_accuracies'].append(val_accuracy)
            train_val_loss_gap = float(val_loss - train_loss)
            train_val_acc_gap = float(train_accuracy - val_accuracy)
            metrics['train_val_loss_gap'].append(train_val_loss_gap)
            metrics['train_val_acc_gap'].append(train_val_acc_gap)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1
                best_state = state

            if wandb_run is not None:
                parameter_l2_norm = jnp.sqrt(sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(state.params)))
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': float(train_loss),
                    'train_accuracy': float(train_accuracy),
                    'val_loss': float(val_loss),
                    'val_accuracy': float(val_accuracy),
                    'train_val_loss_gap': train_val_loss_gap,
                    'train_val_acc_gap': train_val_acc_gap,
                    'best_val_accuracy': float(best_val_accuracy),
                    'best_epoch': best_epoch,
                    'epoch_train_time_s': epoch_train_time,
                    'parameter_l2_norm': float(parameter_l2_norm),
                })
            if (epoch + 1) % save_interval == 0:
                # Save checkpoint as wandb artifact
                checkpoint = {
                    'state': state,
                    'metadata': {
                        'epoch': epoch + 1,
                        'train_loss': float(train_loss),
                        'train_accuracy': float(train_accuracy),
                        'val_loss': float(val_loss),
                        'val_accuracy': float(val_accuracy),
                    }
                }

                checkpoint_manager.save(epoch, checkpoint)

    metrics['train_losses'] = jnp.array(metrics['train_losses'])
    metrics['val_losses'] = jnp.array(metrics['val_losses'])
    metrics['train_accuracies'] = jnp.array(metrics['train_accuracies'])
    metrics['val_accuracies'] = jnp.array(metrics['val_accuracies'])
    metrics['train_val_loss_gap'] = jnp.array(metrics['train_val_loss_gap'])
    metrics['train_val_acc_gap'] = jnp.array(metrics['train_val_acc_gap'])

    print(f"Best validation Accuracy = {best_val_accuracy:.3f} at epoch {best_epoch}")
    print(f"Total training time = {total_train_time:.2f}s, total time (including evaluations) = {time.time() - start_time:.2f}s")

    # Evaluate on test set using the best model
    assert best_state is not None
    test_loss, test_accuracy, test_fpr, test_tpr = evaluate(best_state, test_dataloader, num_classes, tqdm_desc="Testing")
    metrics['test_loss'] = test_loss
    metrics['test_accuracy'] = test_accuracy
    metrics['test_fpr'] = test_fpr
    metrics['test_tpr'] = test_tpr

    if wandb_run is not None:
        wandb.log({
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'best_epoch': best_epoch,
            'total_train_time_s': total_train_time,
            'total_elapsed_time_s': time.time() - start_time,
        })

        checkpoint = {
            'state': state,
            'metadata': {
                'epoch': num_epochs,
                'train_loss': float(train_loss),
                'train_accuracy': float(train_accuracy),
                'val_loss': float(val_loss),
                'val_accuracy': float(val_accuracy),
            }
        }
        checkpoint_manager.save(num_epochs, checkpoint)
        checkpoint_manager.wait_until_finished()
        artifact = wandb.Artifact(f"checkpoint-epoch-{best_epoch}", type="model")
        artifact.add_dir(str(checkpoint_manager.directory))
        wandb_run.log_artifact(artifact)

        wandb_run.finish()

    return metrics

def create_learning_rate_schedule(
        lrs_peak_value: float = 1e-3, 
        lrs_warmup_steps: int = 5_000, 
        lrs_decay_steps: int = 50_000,) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lrs_peak_value,
        warmup_steps=lrs_warmup_steps,
        decay_steps=lrs_decay_steps,
        end_value=0.0
    )
