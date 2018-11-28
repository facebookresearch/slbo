# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from gym.spaces import Box
from slbo.dynamics_model import DynamicsModel
from slbo.envs import BaseBatchedEnv, BaseModelBasedEnv


class VirtualEnv(BaseBatchedEnv):
    _states: np.ndarray

    def __init__(self, model: DynamicsModel, env: BaseModelBasedEnv, n_envs: int, opt_model=False):
        super().__init__()
        self.n_envs = n_envs
        self.observation_space = env.observation_space  # ???

        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0]
        if opt_model:
            self.action_space = Box(low=np.r_[env.action_space.low, np.zeros(dim_state) - 1.],
                                    high=np.r_[env.action_space.high, np.zeros(dim_state) + 1.],
                                    dtype=np.float32)
        else:
            self.action_space = env.action_space

        self._opt_model = opt_model
        self._model = model
        self._env = env

        self._states = np.zeros((self.n_envs, dim_state), dtype=np.float32)

    def _scale_action(self, actions):
        lo, hi = self.action_space.low, self.action_space.high
        return lo + (actions + 1.) * 0.5 * (hi - lo)

    def step(self, actions):
        if self._opt_model:
            actions = actions[..., :self._env.action_space.shape[0]]

        next_states = self._model.eval('next_states', states=self._states, actions=actions)
        rewards, dones = self._env.mb_step(self._states, self._scale_action(actions), next_states)

        self._states = next_states
        return self._states.copy(), rewards, dones, [{} for _ in range(self.n_envs)]

    def reset(self):
        return self.partial_reset(range(self.n_envs))

    def partial_reset(self, indices):
        initial_states = np.array([self._env.reset() for _ in indices])

        self._states = self._states.copy()
        self._states[indices] = initial_states

        return initial_states.copy()

    def set_state(self, states):
        self._states = states.copy()

    def render(self, mode='human'):
        pass
