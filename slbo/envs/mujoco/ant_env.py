# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from rllab.envs.mujoco import ant_env
from rllab.envs.base import Step
from slbo.envs import BaseModelBasedEnv


class AntEnv(ant_env.AntEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # 15
            self.model.data.qvel.flat,  # 14
            # np.clip(self.model.data.cfrc_ext, -1, 1).flat,  # 84
            self.get_body_xmat("torso").flat,  # 9
            self.get_body_com("torso"),  # 9
            self.get_body_comvel("torso"),  # 3
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        comvel = next_states[..., -3:]
        forward_reward = comvel[..., 0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(actions / scaling), axis=-1)
        contact_cost = 0.
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        notdone = np.all([next_states[..., 2] >= 0.2, next_states[..., 2] <= 1.0], axis=0)
        return reward, 1. - notdone

