import glob
import numpy as np
from lower_bounds import lower_bounds


domains_to_envs = {
    'DMC'      : ['acrobot-swingup', 'reacher-hard', 'finger-turn_hard', 'hopper-hop', 'fish-swim', 'cheetah-run', 'walker-run', 'quadruped-run', 'swimmer-swimmer15', 'humanoid-run', 'dog-run'],
    'MuJoCo'   : ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4'],
    'MetaWorld': ['metaworld_button-press-v2', 'metaworld_door-open-v2', 'metaworld_drawer-close-v2', 'metaworld_drawer-open-v2', 'metaworld_peg-insert-side-v2', 'metaworld_pick-place-v2', 'metaworld_push-v2', 'metaworld_reach-v2', 'metaworld_window-open-v2', 'metaworld_window-close-v2', 'metaworld_basketball-v2', 'metaworld_dial-turn-v2', 'metaworld_sweep-into-v2', 'metaworld_hammer-v2', 'metaworld_assembly-v2'],
    'Box2D'    : ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3']
}

def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*.txt', recursive=True)

def load_path_name(path, name):
    paths = listfiles(path, name)
    results = []
    for files in paths:
        result_file = np.loadtxt(files)
        result      = np.array(result_file)
        results.append(result.tolist())

    new_results = []
    min_length = min([len(r) for r in results])
    for r, f in zip(results, files):
        new_results.append(r)
    new_results = [r[:min_length] for r in new_results]
    results     = np.array(new_results)

    return results



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


domains_to_norm_scores = {
    'DMC'      : dmc_max_score,
    'MuJoCo'   : baseline_max_score,
    'MetaWorld': metaworld_max_score,
    'RoboSuite': robosuite_max_score,
    'Box2D'    : box2d_max_score
}




def dmc_actdims(env):
    return {
        'acrobot-swingup': 1,
        'reacher-hard': 2,
        'finger-turn_hard': 2,
        'hopper-hop': 4,
        'fish-swim': 5,
        'cheetah-run': 6,
        'walker-run': 6,
        'quadruped-run': 12,
        'swimmer-swimmer15': 14,
        'humanoid-run': 21,
        'dog-run': 38,
        'humanoid_CMU-run': 56,
    }[env]


def mujoco_actdims(env):
    return {
        'Hopper-v4': 3,
        'HalfCheetah-v4': 6,
        'Walker2d-v4': 6,
        'Ant-v4': 8,
        'Humanoid-v4': 17,
    }[env]


domains_to_actdims = {
    'DMC'      : dmc_actdims,
    'MuJoCo'   : mujoco_actdims,
    'MetaWorld': lambda x: 4,
    'Box2D'    : lambda x: 4
}



for position, domain in enumerate(domains_to_envs):
    for env in domains_to_envs[domain]:
        max_score = domains_to_norm_scores[domain](env)
        min_score = lower_bounds[env]
        actdims   = domains_to_actdims[domain](env)
        print(f"\\texttt{{{env}}} & {actdims} & {max_score:.4g} & {min_score:.4g} \\\ ")

