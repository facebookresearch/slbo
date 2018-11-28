# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from rllab.envs.mujoco import walker2d_env
from rllab.envs.base import Step
from slbo.envs import BaseModelBasedEnv


class Walker2DEnv(walker2d_env.Walker2DEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
            self.get_body_comvel("torso").flat
        ])

    def step(self, action):
        self.forward_dynamics(action)
        forward_reward = self.get_body_comvel("torso")[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 1e-3 * np.sum(np.square(action / scaling))
        alive_bonus = 1.
        reward = forward_reward - ctrl_cost + alive_bonus
        qpos = self.model.data.qpos
        done = not (qpos[0] > 0.8 and qpos[0] < 2.0 and qpos[2] > -1.0 and qpos[2] < 1.0)
        next_obs = self.get_current_obs()
        return Step(next_obs, reward, done)

    def mb_step(self, states, actions, next_states):
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5

        reward_ctrl = -0.001 * np.sum(np.square(actions / scaling), axis=-1)
        reward_fwd = next_states[:, 21]
        alive_bonus = 1.
        rewards = reward_ctrl + reward_fwd + alive_bonus

        dones = ~((next_states[:, 0] > 0.8) &
                  (next_states[:, 0] < 2.0) &
                  (next_states[:, 2] > -1.0) &
                  (next_states[:, 2] < 1.0))
        return rewards, dones
