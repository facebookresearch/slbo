# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf
import numpy as np


class LimitedEntNormal(tf.distributions.Normal):
    def _entropy(self):
        limit = 2.
        lo, hi = (-limit - self._loc) / self._scale / np.sqrt(2), (limit - self._loc) / self._scale / np.sqrt(2)
        return 0.5 * (self._scale.log() + np.log(2 * np.pi) / 2) * (hi.erf() - lo.erf()) + 0.5 * \
            (tf.exp(-hi * hi) * hi - tf.exp(-lo * lo) * lo)

