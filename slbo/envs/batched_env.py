# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from gym import Wrapper
from . import BaseBatchedEnv


class BatchedEnv(BaseBatchedEnv, Wrapper):
    def __init__(self, envs):
        super().__init__(envs[0])
        self.envs = envs
        self.n_envs = len(envs)

    def step(self, actions):

        buf, infos = [], []
        for env, action in zip(self.envs, actions):
            next_state, reward, done, info = env.step(action)
            buf.append((next_state, reward, done))
            infos.append(info)

        return [*(np.array(x) for x in zip(*buf)), infos]

    def reset(self):
        return self.partial_reset(range(self.n_envs))

    def partial_reset(self, indices):
        states = []
        for index in indices:
            states.append(self.envs[index].reset())
        return np.array(states)

    def __repr__(self):
        return f'Batch<{self.n_envs}x {self.env}>'

    def set_state(self, state):
        pass

