import glob
import numpy as np

import utils

domains_to_envs = {
    'DMC'      : ['acrobot-swingup', 'reacher-hard', 'finger-turn_hard', 'hopper-hop', 'fish-swim', 'cheetah-run', 'walker-run', 'quadruped-run', 'swimmer-swimmer15', 'humanoid-run', 'dog-run'],
    'MuJoCo'   : ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4'],
    'MetaWorld': ['metaworld_button-press-v2', 'metaworld_door-open-v2', 'metaworld_drawer-close-v2', 'metaworld_drawer-open-v2', 'metaworld_peg-insert-side-v2', 'metaworld_pick-place-v2', 'metaworld_push-v2', 'metaworld_reach-v2', 'metaworld_window-open-v2', 'metaworld_window-close-v2', 'metaworld_basketball-v2', 'metaworld_dial-turn-v2', 'metaworld_sweep-into-v2', 'metaworld_hammer-v2', 'metaworld_assembly-v2'],
    'Box2D'    : ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3'],
}

DoubleGum_paths = {
    '-0.5': '../../data/camera-ready/continuous/DoubleGum/c-0.5',
    '-0.1': '../../data/camera-ready/continuous/DoubleGum/c-0.1',
    '0.0' : '../../data/camera-ready/continuous/DoubleGum/c0.0',
    '0.1' : '../../data/camera-ready/continuous/DoubleGum/c0.1',
    '0.5' : '../../data/camera-ready/continuous/DoubleGum/c0.5',
}

MoG_paths = {
    'MoG-DDPG': '../../data/camera-ready/continuous/MoG-DDPG',
}

DDPG_paths = {
    'DDPG': '../../data/camera-ready/continuous/DDPG',
    'TD3' : '../../data/camera-ready/continuous/TD3',
}

SAC_paths = {
    'SAC' : '../../data/camera-ready/continuous/SAC/single',
    'TSAC': '../../data/camera-ready/continuous/SAC/twin',
}

XQL_paths = {
    '3' : '../../data/camera-ready/continuous/XQL/single/beta3',
    '4' : '../../data/camera-ready/continuous/XQL/single/beta4',
    '10': '../../data/camera-ready/continuous/XQL/single/beta10',
    '20': '../../data/camera-ready/continuous/XQL/single/beta20',
    '1' : '../../data/camera-ready/continuous/XQL/twin/beta1',
    '2' : '../../data/camera-ready/continuous/XQL/twin/beta2',
    '5' : '../../data/camera-ready/continuous/XQL/twin/beta5',
}

QR_DDPG_paths = {
    'QR-DDPG' : '../../data/camera-ready/continuous/QR-DDPG/single',
    'TQR-DDPG': '../../data/camera-ready/continuous/QR-DDPG/twin',
}

FinerTD3_paths = {
    '0': '../../data/camera-ready/continuous/FinerTD3/pessimism0',
    '1': '../../data/camera-ready/continuous/FinerTD3/pessimism1',
    '2': '../../data/camera-ready/continuous/FinerTD3/pessimism2',
    '3': '../../data/camera-ready/continuous/FinerTD3/pessimism3',
    '4': '../../data/camera-ready/continuous/FinerTD3/pessimism4',
}

all_paths = {
    'MoG-DDPG (untuned)'                   : MoG_paths,
    'best of DDPG/TD3'                     : DDPG_paths,
    'SAC (best w/wo Twin)'                 : SAC_paths,
    r'XQL (best of $\beta$ w/wo Twin)'     : XQL_paths,
    'QR-DDPG (best w/wo Twin)'             : QR_DDPG_paths,
    'FinerTD3 (best pessimism)'            : FinerTD3_paths,
    r'\textbf{DoubleGum, best $c$, (Ours)}': DoubleGum_paths
}

DDPG = {
    'DMC'      : 'DDPG',
    'MuJoCo'   : 'TD3',
    'MetaWorld': 'DDPG',
    'Box2D'    : 'TD3'
}

DoubleGum = {
    'DMC'      : '-0.1',
    'MuJoCo'   : '-0.5',
    'MetaWorld': '0.1',
    'Box2D'    : '-0.1',
}

SAC = {
    'DMC'      : 'SAC',
    'MuJoCo'   : 'TSAC',
    'MetaWorld': 'SAC',
    'Box2D'    : 'TSAC'
}

QR_DDPG = {
    'DMC'      : 'QR-DDPG',
    'MuJoCo'   : 'TQR-DDPG',
    'MetaWorld': 'QR-DDPG',
    'Box2D'    : 'TQR-DDPG'
}

MoG = {
    'DMC'      : 'MoG-DDPG',
    'MuJoCo'   : 'MoG-DDPG',
    'MetaWorld': 'MoG-DDPG',
    'Box2D'    : 'MoG-DDPG'
}

XQL = {
    'DMC' : '3',
    'MuJoCo': '5',
    'MetaWorld': '2',
    'Box2D': '5'
}

FinerTD3 = {
    'DMC': '4',
    'MuJoCo' : '1',
    'MetaWorld': '3',
    'Box2D': '1',
}

pretty_display = {
    r'\textbf{DoubleGum, best $c$, (Ours)}': DoubleGum,
    'MoG-DDPG (untuned)'                   : MoG,
    'best of DDPG/TD3'                     : DDPG,
    'SAC (best w/wo Twin)'                 : SAC,
    'QR-DDPG (best w/wo Twin)'             : QR_DDPG,
    r'XQL (best of $\beta$ w/wo Twin)'     : XQL,
    'FinerTD3 (best pessimism)'            : FinerTD3,
}

algorithms = list(pretty_display.keys())


def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*.txt', recursive=True)


for env in domains_to_envs:
    for position, name in enumerate(domains_to_envs[env]):

        row_means = []
        row_stdevs = []
        for algorithm in algorithms:
            path = all_paths[algorithm][pretty_display[algorithm][env]]
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
