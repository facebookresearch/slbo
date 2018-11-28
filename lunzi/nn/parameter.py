# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf
from lunzi import Tensor


def numpy(self):
    if self.__dict__.get('_numpy_cache', None) is None:
        self._numpy_cache: Tensor = self.eval()
    return self._numpy_cache


def invalidate(self):
    self._numpy_cache = None


# Q: Why not inherit from `tf.Variable`?
# A: Since TensorFlow 1.11, `tf.Variable` has a meta class VariableMetaClass, which overrides `__call__`.
#    And it's `_variable_call` function doesn't explicitly call `tf.Variable` so the return value must
#    be a `tf.Variable`, which makes inheritance impossible.
Parameter = tf.Variable

Parameter.numpy = numpy
Parameter.invalidate = invalidate

