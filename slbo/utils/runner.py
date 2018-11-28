# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from lunzi.dataset import Dataset
from slbo.envs import BaseBatchedEnv
from slbo.policies import BasePolicy
from slbo.utils.dataset import Dataset, gen_dtype
from slbo.v_function import BaseVFunction


class Runner(object):
    _states: np.ndarray  # [np.float]
    _n_steps: np.ndarray
    _returns: np.ndarray

    def __init__(self, env: BaseBatchedEnv, max_steps: int, gamma=0.99, lambda_=0.95, rescale_action=False):
        self.env = env
        self.n_envs = env.n_envs
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_steps = max_steps
        self.rescale_action = rescale_action
        self._dtype = gen_dtype(env, 'state action next_state reward done timeout')

        self.reset()

    def reset(self):
        self.set_state(self.env.reset(), set_env_state=False)

    def set_state(self, states: np.ndarray, set_env_state=True):
        self._states = states.copy()
        if set_env_state:
            self.env.set_state(states)
        self._n_steps = np.zeros(self.n_envs, 'i4')
        self._returns = np.zeros(self.n_envs, 'f8')

    def get_state(self):
        return self._states.copy()

    def run(self, policy: BasePolicy, n_samples: int):
        ep_infos = []
        n_steps = n_samples // self.n_envs
        assert n_steps * self.n_envs == n_samples
        dataset = Dataset(self._dtype, n_samples)

        for T in range(n_steps):
            unscaled_actions = policy.get_actions(self._states)
            if self.rescale_action:
                lo, hi = self.env.action_space.low, self.env.action_space.high
                actions = (lo + (unscaled_actions + 1.) * 0.5 * (hi - lo))
            else:
                actions = unscaled_actions

            next_states, rewards, dones, infos = self.env.step(actions)
            dones = dones.astype(bool)
            self._returns += rewards
            self._n_steps += 1
            timeouts = self._n_steps == self.max_steps

            steps = [self._states.copy(), unscaled_actions, next_states.copy(), rewards, dones, timeouts]
            dataset.extend(np.rec.fromarrays(steps, dtype=self._dtype))

            indices = np.where(dones | timeouts)[0]
            if len(indices) > 0:
                next_states = next_states.copy()
                next_states[indices] = self.env.partial_reset(indices)
                for index in indices:
                    infos[index]['episode'] = {'return': self._returns[index]}
                self._n_steps[indices] = 0
                self._returns[indices] = 0.

            self._states = next_states.copy()
            ep_infos.extend([info['episode'] for info in infos if 'episode' in info])

        return dataset, ep_infos

    def compute_advantage(self, vfn: BaseVFunction, samples: Dataset):
        n_steps = len(samples) // self.n_envs
        samples = samples.reshape((n_steps, self.n_envs))
        use_next_vf = ~samples.done
        use_next_adv = ~(samples.done | samples.timeout)

        next_values = vfn.get_values(samples[-1].next_state)
        values = vfn.get_values(samples.reshape(-1).state).reshape(n_steps, self.n_envs)
        advantages = np.zeros((n_steps, self.n_envs), dtype=np.float32)
        last_gae_lambda = 0

        for t in reversed(range(n_steps)):
            delta = samples[t].reward + self.gamma * next_values * use_next_vf[t] - values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.lambda_ * last_gae_lambda * use_next_adv[t]
            next_values = values[t]
        return advantages.reshape(-1), values.reshape(-1)

