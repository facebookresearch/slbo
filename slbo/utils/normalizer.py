# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List
import numpy as np
import tensorflow as tf
import lunzi.nn as nn
from lunzi.Logger import logger
from lunzi import Tensor
from slbo.utils.np_utils import gaussian_kl


class GaussianNormalizer(nn.Module):
    def __init__(self, name: str, shape: List[int], eps=1e-8, verbose=False):  # batch_size x ...
        super().__init__()

        self.name = name
        self.shape = shape
        self.eps = eps
        self._verbose = verbose

        with self.scope:
            self.op_mean = nn.Parameter(tf.zeros(shape, dtype=tf.float32), name='mean', trainable=False)
            self.op_std = nn.Parameter(tf.ones(shape, dtype=tf.float32), name='std', trainable=False)
            self.op_n = nn.Parameter(tf.zeros([], dtype=tf.int64), name='n', trainable=False)

    def extra_repr(self):
        return f'shape={self.shape}'

    def forward(self, x: Tensor, inverse=False):
        if inverse:
            return x * self.op_std + self.op_mean
        return (x - self.op_mean).div(self.op_std.maximum(self.eps))

    def update(self, samples: np.ndarray):
        old_mean, old_std, old_n = self.op_mean.numpy(), self.op_std.numpy(), self.op_n.numpy()
        samples = samples - old_mean

        m = samples.shape[0]
        delta = samples.mean(axis=0)
        new_n = old_n + m
        new_mean = old_mean + delta * m / new_n
        new_std = np.sqrt((old_std**2 * old_n + samples.var(axis=0) * m + delta**2 * old_n * m / new_n) / new_n)

        kl_old_new = gaussian_kl(new_mean, new_std, old_mean, old_std).sum()
        self.load_state_dict({'op_mean': new_mean, 'op_std': new_std, 'op_n': new_n})

        if self._verbose:
            logger.info("updating Normalizer<%s>, KL divergence = %.6f", self.name, kl_old_new)

    def fast(self, samples: np.ndarray, inverse=False) -> np.ndarray:
        mean, std = self.op_mean.numpy(), self.op_std.numpy()
        if inverse:
            return samples * std + mean
        return (samples - mean) / np.maximum(std, self.eps)


class Normalizers(nn.Module):
    def __init__(self, dim_action: int, dim_state: int):
        super().__init__()
        self.action = GaussianNormalizer('action', [dim_action])
        self.state = GaussianNormalizer('state', [dim_state])
        self.diff = GaussianNormalizer('diff', [dim_state])

    def forward(self):
        pass


