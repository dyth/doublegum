import numpy as np
import glob


def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*.txt', recursive=True)


def load_path_name(path, name):
    '''for continuous control'''
    paths = listfiles(path, name)
    results = []
    for files in paths:
        result_file = np.loadtxt(files)
        result      = np.array(result_file)
        results.append(result.tolist())

    try:
        new_results = []
        min_length = min([len(r) for r in results])
        for r, f in zip(results, files):
            new_results.append(r)
        new_results = [r[:min_length] for r in new_results]
        results     = np.array(new_results)

    except:
        print(path, name)

    return results


domains_to_envs = {
    'DMC'      : ['acrobot-swingup', 'reacher-hard', 'finger-turn_hard', 'hopper-hop', 'fish-swim', 'cheetah-run', 'walker-run', 'quadruped-run', 'swimmer-swimmer15', 'humanoid-run', 'dog-run'],
    'MuJoCo'   : ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4'],
    'MetaWorld': ['metaworld_button-press-v2', 'metaworld_door-open-v2', 'metaworld_drawer-close-v2', 'metaworld_drawer-open-v2', 'metaworld_peg-insert-side-v2', 'metaworld_pick-place-v2', 'metaworld_push-v2', 'metaworld_reach-v2', 'metaworld_window-open-v2', 'metaworld_window-close-v2', 'metaworld_basketball-v2', 'metaworld_dial-turn-v2', 'metaworld_sweep-into-v2', 'metaworld_hammer-v2', 'metaworld_assembly-v2'],
    'Box2D'    : ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3'],
    'Discrete' : ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
}


def suite(env):
    for domain in domains_to_envs:
        if env in domains_to_envs[domain]:
            return domain
