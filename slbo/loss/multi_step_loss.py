# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi import Tensor
from slbo.utils.normalizer import Normalizers


class MultiStepLoss(nn.Module):
    op_train: Tensor
    op_grad_norm: Tensor
    _step: int
    _criterion: nn.Module
    _normalizers: Normalizers
    _model: nn.Module

    def __init__(self, model: nn.Module, normalizers: Normalizers, dim_state: int, dim_action: int,
                 criterion: nn.Module, step=4):
        super().__init__()
        self._step = step
        self._criterion = criterion
        self._model = model
        self._normalizers = normalizers
        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[step, None, dim_state])
            self.op_actions = tf.placeholder(tf.float32, shape=[step, None, dim_action])
            self.op_masks = tf.placeholder(tf.float32, shape=[step, None])
            self.op_next_states_ = tf.placeholder(tf.float32, shape=[step, None, dim_state])

        self.op_loss = self(self.op_states, self.op_actions, self.op_next_states_, self.op_masks)

    def forward(self, states: Tensor, actions: Tensor, next_states_: Tensor, masks: Tensor):
        """
            All inputs have shape [num_steps, batch_size, xxx]
        """

        cur_states = states[0]
        loss = []
        for i in range(self._step):
            next_states = self._model(cur_states, actions[i])
            diffs = next_states - cur_states - next_states_[i] + states[i]
            weighted_diffs = diffs / self._normalizers.diff.op_std.maximum(1e-6)
            loss.append(self._criterion(weighted_diffs, 0, cur_states))

            if i < self._step - 1:
                cur_states = states[i + 1] + masks[i].expand_dims(-1) * (next_states - states[i + 1])

        return tf.add_n(loss) / self._step

    @nn.make_method(fetch='loss')
    def get_loss(self, states, next_states_, actions, masks): pass

    def build_backward(self, lr: float, weight_decay: float, max_grad_norm=2.):
        loss = self.op_loss.reduce_mean(name='Loss')

        optimizer = tf.train.AdamOptimizer(lr)
        params = self._model.parameters()
        regularization = weight_decay * tf.add_n([tf.nn.l2_loss(t) for t in params], name='regularization')

        grads_and_vars = optimizer.compute_gradients(loss + regularization, var_list=params)
        print([var.name for grad, var in grads_and_vars])
        clip_grads, op_grad_norm = tf.clip_by_global_norm([grad for grad, _ in grads_and_vars], max_grad_norm)
        clip_grads_and_vars = [(grad, var) for grad, (_, var) in zip(clip_grads, grads_and_vars)]
        self.op_train = optimizer.apply_gradients(clip_grads_and_vars)
        self.op_grad_norm = op_grad_norm
