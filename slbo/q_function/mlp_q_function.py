# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List
import tensorflow as tf
from . import BaseQFunction
import lunzi.nn as nn
from slbo.utils.multi_layer_perceptron import MultiLayerPerceptron


class MLPQFunction(MultiLayerPerceptron, BaseQFunction):
    def __init__(self, dim_state: int, dim_action: int, hidden_states: List[int]):
        super().__init__((dim_state + dim_action, *hidden_states, 1), squeeze=True)
        self._dim_state = dim_state
        self._dim_action = dim_action

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, [None, dim_state])
            self.op_actions = tf.placeholder(tf.float32, [None, dim_action])

        self.op_Q = self.forward(self.op_states, self.op_actions)

    @nn.make_method(fetch='Q')
    def get_q(self, states, actions): pass
