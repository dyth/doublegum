import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams.update({'font.size': 20})

# latex text rendering, from https://stackoverflow.com/a/8384685
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']

import utils

fontsize = 25

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


def make_legend(ax, algorithms=None):
    color_palette   = sns.color_palette('colorblind', n_colors=len(algorithms))
    base_algorithms = [r'\textbf{DoubleGum, $c=-0.1$ (Ours)}', 'MoG-DDPG', 'DDPG', 'SAC', 'XQL', 'QR-DDPG', 'FinerTD3']
    colors          = dict(zip(base_algorithms, color_palette))
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for algorithm in algorithms:
        if 'T' in algorithm:
            linestyle = 'dashed'
        else:
            linestyle = 'solid'

        alg_to_base = {
            r'\textbf{DoubleGum, $c=-0.1$ (Ours)}': r'\textbf{DoubleGum, $c=-0.1$ (Ours)}',
            'DDPG'                                : 'DDPG',
            'TD3'                                 : 'DDPG',
            'MoG-DDPG'                            : 'MoG-DDPG',
            'SAC'                                 : 'SAC',
            'Twin-SAC'                            : 'SAC',
            'Twin-XQL'                            : 'XQL',
            'QR-DDPG'                             : 'QR-DDPG',
            'TQR-DDPG'                            : 'QR-DDPG',
            'FinerTD3'                            : 'FinerTD3'
        }
        plt.plot(0, 0, color=colors[alg_to_base[algorithm]], label=algorithm, linestyle=linestyle)
    leg = ax.legend(loc='center left', ncol=2, fontsize=fontsize)

    # from https://stackoverflow.com/a/48296983
    for legobj in leg.legend_handles:
        legobj.set_linewidth(3.0)


def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*.txt', recursive=True)


for env in domains_to_envs:
    width  = 6
    height = 1 + len(domains_to_envs[env]) // width
    fig, axes = plt.subplots(height, width, figsize=(22, .5 + height * 4))
    for position, name in enumerate(domains_to_envs[env]):
        x, y = int(position / width), position % width

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

            if height == 1:
                current_axis = axes[y]
            else:
                current_axis = axes[x, y]


            color_palette          = sns.color_palette('colorblind', n_colors=len(algorithms))
            base_algorithms        = [r'\textbf{DoubleGum, $c=-0.1$ (Ours)}', 'MoG-DDPG', 'DDPG', 'SAC', 'XQL', 'QR-DDPG', 'FinerTD3']
            colors                 = dict(zip(base_algorithms, color_palette))
            inverse_pretty_display = {v: k for k, v in pretty_display.items()}
            algorithm              = inverse_pretty_display[path]

            if 'T' in algorithm:
                linestyle = 'dashed'
            else:
                linestyle = 'solid'

            alg_to_base = {
                r'\textbf{DoubleGum, $c=-0.1$ (Ours)}': r'\textbf{DoubleGum, $c=-0.1$ (Ours)}',
                'DDPG': 'DDPG',
                'TD3': 'DDPG',
                'MoG-DDPG': 'MoG-DDPG',
                'SAC': 'SAC',
                'Twin-SAC': 'SAC',
                'Twin-XQL': 'XQL',
                'QR-DDPG': 'QR-DDPG',
                'TQR-DDPG': 'QR-DDPG',
                'FinerTD3': 'FinerTD3'
            }

            color = colors[alg_to_base[algorithm]]
            current_axis.plot(index, means, color=color, linewidth=2 if 'DoubleGum' in path else 1.2, linestyle=linestyle)
            current_axis.set_title(name.replace('metaworld_', ''), fontsize=fontsize)
            current_axis.fill_between(index, np.array(means)-stdev, np.array(means)+stdev, alpha=alpha, color=color)

        if height == 1:
            utils._decorate_axis(axes[y], landscape=False)
        else:
            utils._decorate_axis(axes[x, y], landscape=False)

    if env == 'Box2D':
        # from https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
        gs = axes[2].get_gridspec()
        axes[2].remove()
        axes[3].remove()
        axes[4].remove()
        axes[5].remove()
        bigaxes = fig.add_subplot(gs[2:5])
        make_legend(bigaxes, algorithms=algorithms)

    else:
        if height == 1:
            for i in range(y+1, width):
                axes[i].remove()
        else:
            for i in range(y+1, width):
                axes[x, i].remove()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Timesteps (in millions)', fontsize=30)
    plt.ylabel('IQM\n', fontsize=30)
    plt.tight_layout()
    utils.save_fig(plt, f'figs17/{env}')

plt.show()
