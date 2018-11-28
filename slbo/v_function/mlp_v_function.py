# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf
from baselines.common.tf_util import normc_initializer
from slbo.utils.multi_layer_perceptron import MultiLayerPerceptron
import lunzi.nn as nn
from . import BaseVFunction


class MLPVFunction(BaseVFunction, nn.Module):
    def __init__(self, dim_state, hidden_sizes, normalizer=None):
        super().__init__()
        self.mlp = MultiLayerPerceptron((dim_state, *hidden_sizes, 1), activation=nn.Tanh, squeeze=True,
                                        weight_initializer=normc_initializer(1.), build=False)
        self.normalizer = normalizer
        self.op_states = tf.placeholder(tf.float32, shape=[None, dim_state])
        self.op_values = self.forward(self.op_states)

    def forward(self, states):
        states = self.normalizer(states)
        return self.mlp(states)

    @nn.make_method(fetch='values')
    def get_values(self, states): pass

