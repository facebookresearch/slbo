# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import abc
import gym
from slbo.utils.dataset import Dataset, gen_dtype
from lunzi.Logger import logger


class BaseBatchedEnv(gym.Env, abc.ABC):
    # thought about using `@property @abc.abstractmethod` here but we don't need explicit `@property` function here.
    n_envs: int

    @abc.abstractmethod
    def step(self, actions):
        pass

    def reset(self):
        return self.partial_reset(range(self.n_envs))

    @abc.abstractmethod
    def partial_reset(self, indices):
        pass

    def set_state(self, state):
        logger.warning('`set_state` is not implemented')


class BaseModelBasedEnv(gym.Env, abc.ABC):
    @abc.abstractmethod
    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        raise NotImplementedError

    def verify(self, n=2000, eps=1e-4):
        dataset = Dataset(gen_dtype(self, 'state action next_state reward done'), n)
        state = self.reset()
        for _ in range(n):
            action = self.action_space.sample()
            next_state, reward, done, _ = self.step(action)
            dataset.append((state, action, next_state, reward, done))

            state = next_state
            if done:
                state = self.reset()

        rewards_, dones_ = self.mb_step(dataset.state, dataset.action, dataset.next_state)
        diff = dataset.reward - rewards_
        l_inf = np.abs(diff).max()
        logger.info('rewarder difference: %.6f', l_inf)

        assert np.allclose(dones_, dataset.done)
        assert l_inf < eps

    def seed(self, seed: int = None):
        pass

