import jax
import jax.numpy as jnp


class ParityDataset:
    """Dataset of all 2^n bitstrings with parity labels."""

    def __init__(self, n: int = 4, seed: int = 42):
        self.n = n
        self.seed = seed
        self.num_samples = 2 ** n - 1  # exclude the all-zero string

        # Generate all bitstrings of length n (skip the all-zero string)
        indices = jnp.arange(1, 2 ** n)
        X = jnp.array(
            [[(int(idx) >> (n - 1 - b)) & 1 for b in range(n)] for idx in indices],
            dtype=jnp.int32,
        )
        print("Generated input data X with shape:", X.shape)
        Y = jnp.sum(X, axis=1) % 2  # parity: 1 if odd number of 1s, else 0

        self.X = X
        self.Y = Y

    def __len__(self):
        return self.num_samples

    def __repr__(self):
        return f"ParityDataset(n={self.n}, num_samples={self.num_samples})"

    def get_random_split(self, training_frac: float = 0.8, test_frac: float = 0.0):
        if training_frac + test_frac > 1:
            raise ValueError("Fractions must sum to at most 1")
        n = len(self)
        training_length = int(training_frac * n)
        test_length = int(test_frac * n)
        validation_length = n - training_length - test_length

        key = jax.random.PRNGKey(self.seed)
        indices = jax.random.permutation(key, n)

        train_idx = indices[:training_length]
        if validation_length > 0:
            val_idx = indices[training_length : training_length + validation_length]
            test_idx = indices[training_length + validation_length :]
            return {
                "training": (self.X[train_idx], self.Y[train_idx]),
                "validation": (self.X[val_idx], self.Y[val_idx]),
                "test": (self.X[test_idx], self.Y[test_idx]),
            }
        else:
            test_idx = indices[training_length:]
            return {
                "training": (self.X[train_idx], self.Y[train_idx]),
                "test": (self.X[test_idx], self.Y[test_idx]),
            }

if __name__ == "__main__":
    dataset = ParityDataset(n=4)
    print(dataset)
    splits = dataset.get_random_split(training_frac=0.8, test_frac=0.2)
    print("Training set size:", len(splits["training"][0]))
    print("Test set size:", len(splits["test"][0]))