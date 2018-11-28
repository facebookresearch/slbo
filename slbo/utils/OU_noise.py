# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from slbo.policies import BasePolicy


class OUNoise(object):
    _policy: BasePolicy

    def __init__(self, action_space, mu=0.0, theta=0.15, sigma=0.3, shape=None):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = action_space
        self._state = None
        if shape:
            self.shape = shape
        else:
            self.shape = action_space.shape

        self.reset()

    def reset(self):
        self._state = np.ones(self.shape) * self.mu

    def next(self):
        delta = self.theta * (self.mu - self._state) + self.sigma * np.random.randn(*self._state.shape)
        self._state = self._state + delta
        return self._state

    def get_actions(self, states):
        return self._policy.get_actions(states) + self.next()

    def make(self, policy: BasePolicy):
        self._policy = policy
        return self

