# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf
import numpy as np
import lunzi.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, blocks, activation=nn.ReLU, squeeze=False, weight_initializer=None, build=True):
        super().__init__()

        self._blocks = blocks
        if build:
            self.op_inputs = tf.placeholder(tf.float32, [None, self._blocks[0]])

        with self.scope:
            kwargs = {}
            if weight_initializer is not None:
                kwargs['weight_initializer'] = weight_initializer
            layers = []
            for in_features, out_features in zip(blocks[:-1], blocks[1:]):
                if layers:
                    layers.append(activation())
                layers.append(nn.Linear(in_features, out_features, **kwargs))
            if squeeze:
                layers.append(nn.Squeeze(axis=1))
            self.net = nn.Sequential(*layers)

        self._squeeze = squeeze
        self._activation = activation

        if build:
            self.build()

    def build(self):
        self.op_outputs = self.forward(self.op_inputs)

    def forward(self, *inputs):
        if len(inputs) > 1:
            inputs = tf.concat(inputs, axis=-1)
        else:
            inputs = inputs[0]
        return self.net(inputs)

    def fast(self, *inputs):
        return self.net.fast(np.concatenate(inputs, axis=-1))

    def clone(self):
        return MultiLayerPerceptron(self._blocks, self._activation, self._squeeze)

    def extra_repr(self):
        return f'activation = {self._activation}, blocks = {self._blocks}, squeeze = {self._squeeze}'
