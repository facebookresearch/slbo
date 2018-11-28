# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from rllab.envs.mujoco import swimmer_env
from slbo.envs import BaseModelBasedEnv


class SwimmerEnv(swimmer_env.SwimmerEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # 5
            self.model.data.qvel.flat,  # 5
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso"),  # 3
        ]).reshape(-1)

    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(actions / scaling), axis=-1)
        forward_reward = next_states[:, -3]
        reward = forward_reward - ctrl_cost
        return reward, np.zeros_like(reward, dtype=np.bool)
