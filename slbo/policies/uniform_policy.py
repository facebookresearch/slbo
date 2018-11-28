# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from . import BasePolicy


class UniformPolicy(BasePolicy):
    def __init__(self, dim_action):
        self.dim_action = dim_action

    def get_actions(self, states):
        return np.random.uniform(-1., 1., states.shape[:-1] + (self.dim_action,))
