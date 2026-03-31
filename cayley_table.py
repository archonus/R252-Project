import operator

import jax
import jax.numpy as jnp
import numpy as np


class CayleyTableDataset:
    def __init__(self, p: int = 5, seed: int = 42, op=operator.add):
        self.p = p
        self.seed = seed
        self.op = op
        key = jax.random.PRNGKey(seed)
        self.pi = jax.random.permutation(key, p)
        X = []
        Y = []
        for x1 in range(p):
            for x2 in range(p):
                X.append([self.pi[x1], self.pi[x2]])
                Y.append(self.pi[self.op(x1, x2) % p])
        self.X = jnp.array(X)
        self.Y = jnp.array(Y)
        self.num_bits = int(np.ceil(np.log2(self.p))) if self.p > 1 else 1
        self.X_enc, self.Y_enc = self.get_basis_encoded()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __repr__(self) -> str:
        return f'CayleyTableDataset(N={self.p}, seed={self.seed}, op={self.op.__name__})\n{self.get_table()}'

    def get_table(self):
        table = np.zeros((self.p, self.p), dtype=int)
        for x1 in range(self.p):
            for x2 in range(self.p):
                table[self.pi[x1], self.pi[x2]] = self.pi[self.op(x1, x2) % self.p]
        return table
        

    def get_basis_encoded(self):
        """Return basis-encoded X and Y as binary bitstrings.

        Each integer is encoded as its binary representation using
        num_bits = ceil(log2(N)) bits, suitable for use with qml.BasisState.

        Returns:
            X_enc: jnp.array of shape (N*N, 2, num_bits) — each of the two
                   inputs encoded as a separate binary vector.
            Y_enc: jnp.array of shape (N*N, num_bits) — output encoded as
                   a binary vector.
        """
        
        num_bits = self.num_bits
        def to_bits(x):
            return jnp.array([(int(x) >> (num_bits - 1 - b)) & 1 for b in range(num_bits)], dtype=jnp.int32)

        X_enc = jnp.array([[to_bits(self.X[i, 0]), to_bits(self.X[i, 1])] for i in range(len(self))])
        Y_enc = jnp.array([to_bits(self.Y[i]) for i in range(len(self))])
        return X_enc, Y_enc
    
    def get_random_split(self, training_frac: float = 0.8, test_frac: float = 0.1):
        if training_frac + test_frac > 1:
            raise ValueError('Fraction must sum to 1')
        n = len(self)
        training_length = int(training_frac * n)
        test_length = int(test_frac * n)
        validation_length = n - training_length - test_length

        key = jax.random.PRNGKey(self.seed)
        indices = jax.random.permutation(key, n)

        train_idx = indices[:training_length]
        if validation_length > 0:
            val_idx = indices[training_length:training_length + validation_length]
            test_idx = indices[training_length + validation_length:]
            return {
                'training': (self.X[train_idx], self.Y[train_idx]),
                'validation': (self.X[val_idx], self.Y[val_idx]),
                'test': (self.X[test_idx], self.Y[test_idx]),
            }
        else:
            test_idx = indices[training_length:]
            return {
                'training': (self.X[train_idx], self.Y[train_idx]),
                'test': (self.X[test_idx], self.Y[test_idx]),
            }


    def get_random_split_encoded(self, training_frac: float = 0.8, test_frac: float = 0.1):
        if training_frac + test_frac > 1:
            raise ValueError('Fraction must sum to 1')
        n = len(self)
        training_length = int(training_frac * n)
        test_length = int(test_frac * n)
        validation_length = n - training_length - test_length

        key = jax.random.PRNGKey(self.seed)
        indices = jax.random.permutation(key, n)

        train_idx = indices[:training_length]
        if validation_length > 0:
            val_idx = indices[training_length:training_length + validation_length]
            test_idx = indices[training_length + validation_length:]
            return {
                'training': (self.X_enc[train_idx], self.Y_enc[train_idx]),
                'validation': (self.X_enc[val_idx], self.Y_enc[val_idx]),
                'test': (self.X_enc[test_idx], self.Y_enc[test_idx]),
            }
        else:
            test_idx = indices[training_length:]
            return {
                'training': (self.X_enc[train_idx], self.Y_enc[train_idx]),
                'test': (self.X_enc[test_idx], self.Y_enc[test_idx]),
            }
