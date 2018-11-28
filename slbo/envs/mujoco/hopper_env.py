# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from rllab.envs.mujoco import hopper_env
from rllab.envs.base import Step
from slbo.envs import BaseModelBasedEnv


class HopperEnv(hopper_env.HopperEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # 6
            self.model.data.qvel.flat,  # 6
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso"),  # 3
        ])

    def mb_step(self, states, actions, next_states):
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        vel = next_states[:, -3]
        reward = vel + self.alive_coeff - 0.5 * self.ctrl_cost_coeff * np.sum(np.square(actions / scaling), axis=-1)

        done = ~((next_states[:, 3:12] < 100).all(axis=-1) &
                 (next_states[:, 0] > 0.7) &
                 (np.abs(next_states[:, 2]) < 0.2))
        return reward, done
