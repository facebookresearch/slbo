# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf
import numpy as np
from .module import Module
from .parameter import Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True, weight_initializer=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weight_initializer is None:
            init_range = tf.sqrt(6.0 / (in_features + out_features))
            weight_initializer = tf.random_uniform_initializer(-init_range, init_range, dtype=tf.float32)

        self.use_bias = bias
        with self.scope:
            self.op_input = tf.placeholder(dtype=tf.float32, shape=[None, in_features], name='input')
            self.weight = Parameter(weight_initializer([in_features, out_features]), name='weight')
            if bias:
                self.bias = Parameter(tf.zeros([out_features], dtype=tf.float32), name='bias')

        self.op_output = self(self.op_input)

    def forward(self, x):
        shape = x.get_shape().as_list()
        if len(shape) > 2:
            y = tf.tensordot(x, self.weight, [[len(shape) - 1], [0]])
        else:
            y = x.matmul(self.weight)
        if self.use_bias:
            y = y + self.bias
        return y

    def fast(self, x):
        x = x.dot(self.weight.numpy())
        if self.use_bias:
            x = x + self.bias.numpy()
        return x

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}'


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[i] = module

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    def fast(self, x):
        for module in self._modules.values():
            x = module.fast(x)
        return x


class ReLU(Module):
    def forward(self, x):
        return tf.nn.relu(x)

    def fast(self, x: np.ndarray):
        return np.maximum(x, 0)


class Tanh(Module):
    def forward(self, x):
        return tf.nn.tanh(x)

    def fast(self, x: np.ndarray):
        return np.tanh(x)


class Squeeze(Module):
    def __init__(self, axis=None):
        super().__init__()
        self._axis = axis

    def forward(self, x):
        return x.squeeze(axis=self._axis)

    def fast(self, x):
        return x.squeeze(axis=self._axis)
