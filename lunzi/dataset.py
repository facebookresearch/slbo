# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np


class Dataset(np.recarray):
    """
        Overallocation can be supported, by making examinations before
        each `append` and `extend`.
    """

    @staticmethod
    def fromarrays(array_lists, dtype):
        array = np.rec.fromarrays(array_lists, dtype=dtype)
        ret = Dataset(dtype, len(array))
        ret.extend(array)
        return ret

    def __init__(self, dtype, max_size, verbose=False):
        super().__init__()
        self.max_size = max_size
        self._index = 0
        self._buf_size = 0
        self._len = 0

        self.resize(max_size)
        self._buf_size = max_size

    def __new__(cls, dtype, max_size):
        return np.recarray.__new__(cls, max_size, dtype=dtype)

    def size(self):
        return self._len

    def reserve(self, size):
        cur_size = max(self._buf_size, 1)
        while cur_size < size:
            cur_size *= 2
        if cur_size != self._buf_size:
            self.resize(cur_size)

    def clear(self):
        self._index = 0
        self._len = 0
        return self

    def append(self, item):
        self[self._index] = item
        self._index = (self._index + 1) % self.max_size
        self._len = min(self._len + 1, self.max_size)
        return self

    def extend(self, items):
        n_new = len(items)
        if n_new > self.max_size:
            items = items[-self.max_size:]
            n_new = self.max_size

        n_tail = self.max_size - self._index
        if n_new <= n_tail:
            self[self._index:self._index + n_new] = items
        else:
            n_head = n_new - n_tail
            self[self._index:] = items[:n_tail]
            self[:n_head] = items[n_tail:]

        self._index = (self._index + n_new) % self.max_size
        self._len = min(self._len + n_new, self.max_size)
        return self

    def sample(self, size, indices=None):
        if indices is None:
            indices = np.random.randint(0, self._len, size=size)
        return self[indices]

    def iterator(self, batch_size):
        indices = np.arange(self._len, dtype=np.int32)
        np.random.shuffle(indices)
        index = 0
        while index + batch_size <= self._len:
            end = index + batch_size
            yield self[indices[index:end]]
            index = end
