# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from slbo.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from slbo.envs.mujoco.walker2d_env import Walker2DEnv
from slbo.envs.mujoco.humanoid_env import HumanoidEnv
from slbo.envs.mujoco.ant_env import AntEnv
from slbo.envs.mujoco.hopper_env import HopperEnv
from slbo.envs.mujoco.swimmer_env import SwimmerEnv


def make_env(id: str):
    envs = {
        'HalfCheetah-v2': HalfCheetahEnv,
        'Walker2D-v2': Walker2DEnv,
        'Humanoid-v2': HumanoidEnv,
        'Ant-v2': AntEnv,
        'Hopper-v2': HopperEnv,
        'Swimmer-v2': SwimmerEnv,
    }
    env = envs[id]()
    if not hasattr(env, 'reward_range'):
        env.reward_range = (-np.inf, np.inf)
    if not hasattr(env, 'metadata'):
        env.metadata = {}
    env.seed(np.random.randint(2**60))
    return env
