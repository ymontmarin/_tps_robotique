import numpy as np
import pickle as pkl
from utils.datastructures.storage import Storage


class PathTree:

    @classmethod
    def load(cls, path):
        inst = cls(Storage.load(str(path) + '_storage.pkl'))
        with open(str(path) + '_tree.pkl', 'rb') as f:
            a = pkl.load(f)

        n = inst.storage.n
        inst.parent[:n] = a['parent']
        inst.cost[:n] = a['cost']
        inst.depth[:n] = a['depth']

        return inst

    def __init__(self, storage):
        self.storage = storage
        self.parent = np.zeros(storage.N, dtype=np.intp)
        self.cost = np.zeros(storage.N, dtype=float)
        self.depth = np.zeros(storage.N, dtype=int)

    def update_link(self, q_idx, parent_idx, c=1.):
        self.parent[q_idx] = parent_idx
        self.depth[q_idx] = self.depth[parent_idx] + 1
        self.cost[q_idx] = self.cost[parent_idx] + c

    def get_edges(self):
        # TODO use yielding to avoid data overcreation
        res = np.zeros((self.storage.n - 1, 2, self.storage.dim), dtype=np.float)
        res[:, 0, :] = self.storage.data[1:self.storage.n, :]
        res[:, 1, :] = self.storage.data[self.parent[1:self.storage.n], :]

        costs = self.cost[1:self.storage.n]
        return res, costs

    def get_path(self):
        # TODO use yielding to avoid data overcreation
        i = self.storage.n - 1
        len_path = self.depth[i] + 1
        res = np.zeros((len_path, self.storage.dim))
        j = len_path
        while not i == 0:
            j -= 1
            res[j] = self.storage.data[i]
            i = self.parent[i]
        res[0] = self.storage.data[0]
        return res

    def save(self, path):
        n = self.storage.n
        self.storage.save(str(path) + '_storage.pkl')
        with open(str(path) + '_tree.pkl', 'wb') as f:
            pkl.dump({
                'parent': self.parent[:n],
                'cost': self.cost[:n],
                'depth': self.depth[:n],
            }, f)

    def get_estimated_start_goal(self):
        return self.storage.data[0], self.storage.data[self.storage.n - 1]
