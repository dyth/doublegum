import numpy as np
from utils import load_path_name


baseline_path_lists = [
    '../../data/camera-ready/continuous/DDPG',
    '../../data/camera-ready/continuous/FinerTD3',
    '../../data/camera-ready/continuous/MoG-DDPG',
    '../../data/camera-ready/continuous/QR-DDPG',
    '../../data/camera-ready/continuous/SAC',
    '../../data/camera-ready/continuous/TD3',
    '../../data/camera-ready/continuous/XQL',
]


def dmc_max_score(env):
    return 1000.


def baseline_max_score(env):
    # read all baseline files and select maximum value at last timestep
    maxes = []
    for bpl in baseline_path_lists:
        results = load_path_name(bpl, env)
        maxes.append(np.amax(results))
    return max(maxes)


def metaworld_max_score(env):
    return 10000.


def robosuite_max_score(env):
    return 500.


def box2d_max_score(env):
    # https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/box2d/bipedal_walker.py#L109-L110
    if env == 'BipedalWalker-v3':
        return 300.
    elif env == 'BipedalWalkerHardcore-v3':
        return 300.
    return 0.


upper_bounds = {
    'DMC'      : dmc_max_score,
    'MuJoCo'   : baseline_max_score,
    'MetaWorld': metaworld_max_score,
    'RoboSuite': robosuite_max_score,
    'Box2D'    : box2d_max_score,
}
