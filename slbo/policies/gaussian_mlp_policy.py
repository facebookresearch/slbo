# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List
import tensorflow as tf
import numpy as np
from lunzi import Tensor
from lunzi import nn
from baselines.common.tf_util import normc_initializer
from slbo.utils.truncated_normal import LimitedEntNormal
from . import BasePolicy
from slbo.utils.normalizer import GaussianNormalizer


class GaussianMLPPolicy(nn.Module, BasePolicy):
    op_states: Tensor

    def __init__(self, dim_state: int, dim_action: int, hidden_sizes: List[int], normalizer: GaussianNormalizer,
                 init_std=1.):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = hidden_sizes
        self.init_std = init_std
        self.normalizer = normalizer
        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[None, dim_state], name='states')
            self.op_actions_ = tf.placeholder(tf.float32, shape=[None, dim_action], name='actions')

            layers = []
            # note that the placeholder has size 105.
            all_sizes = [dim_state, *self.hidden_sizes]
            for i, (in_features, out_features) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
                layers.append(nn.Linear(in_features, out_features, weight_initializer=normc_initializer(1)))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(all_sizes[-1], dim_action, weight_initializer=normc_initializer(0.01)))
            self.net = nn.Sequential(*layers)

            self.op_log_std = nn.Parameter(
                tf.constant(np.log(self.init_std), shape=[self.dim_action], dtype=tf.float32), name='log_std')

        self.distribution = self(self.op_states)
        self.op_actions = self.distribution.sample()
        self.op_actions_mean = self.distribution.mean()
        self.op_actions_std = self.distribution.stddev()
        self.op_nlls_ = -self.distribution.log_prob(self.op_actions_).reduce_sum(axis=1)

        self.register_callable('[states] => [actions]', self.fast)

    def forward(self, states):
        states = self.normalizer(states)
        actions_mean = self.net(states)
        distribution = LimitedEntNormal(actions_mean, self.op_log_std.exp())

        return distribution

    @nn.make_method(fetch='actions')
    def get_actions(self, states): pass

    def fast(self, states, use_log_prob=False):
        states = self.normalizer.fast(states)
        actions_mean = self.net.fast(states)
        noise = np.random.randn(*actions_mean.shape)
        actions = actions_mean + noise * np.exp(self.op_log_std.numpy())
        if use_log_prob:
            log_prob = -noise**2 / 2 - np.log(2 * np.pi) / 2 - self.op_log_std.numpy()
            return actions, log_prob.sum(axis=1)
        return actions

    def clone(self):
        return GaussianMLPPolicy(self.dim_state, self.dim_action, self.hidden_sizes, self.normalizer, self.init_std)
