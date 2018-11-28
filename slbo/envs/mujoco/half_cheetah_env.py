# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from rllab.envs.mujoco import half_cheetah_env
from slbo.envs import BaseModelBasedEnv


class HalfCheetahEnv(half_cheetah_env.HalfCheetahEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # 9
            self.model.data.qvel.flat,  # 9
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso").flat,  # 3
        ])

    def mb_step(self, states, actions, next_states):
        actions = np.clip(actions, *self.action_bounds)
        reward_ctrl = -0.05 * np.sum(np.square(actions), axis=-1)
        reward_fwd = next_states[..., 21]
        return reward_ctrl + reward_fwd, np.zeros_like(reward_fwd, dtype=np.bool)
