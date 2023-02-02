import numpy as np
import pickle as pkl


class Storage:

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            a = pkl.load(f)

        inst = cls(a['N'], a['dim'])

        n = a['n']
        inst.n = n

        inst.data[:n] = a['data']

        return inst

    def __init__(self, N, dim):
        self.N = N
        self.dim = dim
        self.n = np.intp(0)
        self.data = np.zeros((N, dim), dtype=float)

    def add_point(self, p):
        assert not self.is_full
        self.data[self.n] = p
        self.n += 1
        return self.n - 1

    def remove_last(self):
        assert self.n
        self.n -= 1

    def __getitem__(self, idx):
        # assert idx < self.n
        return self.data[idx]

    def __len__(self):
        return self.n

    @property
    def ndarray(self):
        return self.data[:self.n]

    @property
    def is_full(self):
        return self.n == self.N

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump({
                'N': self.N,
                'dim': self.dim,
                'n': self.n,
                'data': self.data[:self.n]
            }, f)
