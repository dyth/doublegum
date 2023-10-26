import glob
import numpy as np

import utils

domains_to_envs = {
    'DMC'      : ['acrobot-swingup', 'reacher-hard', 'finger-turn_hard', 'hopper-hop', 'fish-swim', 'cheetah-run', 'walker-run', 'quadruped-run', 'swimmer-swimmer15', 'humanoid-run', 'dog-run'],
    'MuJoCo'   : ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4'],
    'MetaWorld': ['metaworld_button-press-v2', 'metaworld_door-open-v2', 'metaworld_drawer-close-v2', 'metaworld_drawer-open-v2', 'metaworld_peg-insert-side-v2', 'metaworld_pick-place-v2', 'metaworld_push-v2', 'metaworld_reach-v2', 'metaworld_window-open-v2', 'metaworld_window-close-v2', 'metaworld_basketball-v2', 'metaworld_dial-turn-v2', 'metaworld_sweep-into-v2', 'metaworld_hammer-v2', 'metaworld_assembly-v2'],
    'Box2D'    : ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3'],
}

path_lists = [
    # benchmark
    '../../data/camera-ready/continuous/DoubleGum/c-0.1',
    '../../data/camera-ready/continuous/MoG-DDPG',
    '../../data/camera-ready/continuous/DDPG',
    '../../data/camera-ready/continuous/TD3',
    '../../data/camera-ready/continuous/SAC/twin',
    '../../data/camera-ready/continuous/XQL/twin/beta5',
    '../../data/camera-ready/continuous/QR-DDPG/single',
    '../../data/camera-ready/continuous/FinerTD3/pessimism1',
]

pretty_display = {
    r'\textbf{DoubleGum, $c=-0.1$ (Ours)}': '../../data/camera-ready/continuous/DoubleGum/c-0.1',
    'MoG-DDPG'                            : '../../data/camera-ready/continuous/MoG-DDPG',
    'DDPG'                                : '../../data/camera-ready/continuous/DDPG',
    'TD3'                                 : '../../data/camera-ready/continuous/TD3',
    'Twin-SAC'                            : '../../data/camera-ready/continuous/SAC/twin',
    'Twin-XQL'                            : '../../data/camera-ready/continuous/XQL/twin/beta5',
    'QR-DDPG'                             : '../../data/camera-ready/continuous/QR-DDPG/single',
    'FinerTD3'                            : '../../data/camera-ready/continuous/FinerTD3/pessimism1',
}

algorithms = list(pretty_display.keys())

def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*.txt', recursive=True)


for env in domains_to_envs:
    width  = 6
    height = 1 + len(domains_to_envs[env]) // width
    for position, name in enumerate(domains_to_envs[env]):

        row_means = []
        row_stdevs = []
        for path in path_lists:
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

            proportiontocut = 0.25
            samples         = 12
            cut             = int(proportiontocut * samples)

            results.sort(0)
            stdev = results.std(0)

            results = results[cut:-cut]

            means = results.mean(0)
            index = (np.arange(results.shape[1])) / 100
            index = index.tolist()

            phi = (np.sqrt(5) - 1) / 2
            alpha = phi / len(path_lists)

            # current_axis.plot(index, means, color=color, linewidth=2 if 'DoubleGum' in path else 1.2, linestyle=linestyle)

            row_means.append(means[-1])
            row_stdevs.append(stdev[-1])

        biggest = max(row_means)
        row = f"\\texttt{{{name.replace('_metaworld', '')}}} "
        for rm, rs in zip(row_means, row_stdevs):
            if rm == biggest:
                row += f"& \\textbf{{{rm:.4g} $\pm$ {rs:.4g}}} "
            else:
                row += f"& {rm:.4g} $\pm$ {rs:.4g} "
        row += "\\\ "
        print(row)
