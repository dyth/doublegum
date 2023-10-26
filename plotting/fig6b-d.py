import glob
import re
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams.update({'font.size': 20})
from scipy.stats import logistic

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
    f'../../data/camera-ready/continuous/DoubleGum/c-0.1/td',
]

def make_legend(ax, algorithms=None):
    colors = sns.color_palette('colorblind')
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.plot(0, 0, color=colors[1], label='Homo-Normal')
    plt.plot(0, 0, color='black', label=r'\textbf{Hetero-Logistic (expected)}')
    plt.plot(0, 0, color=colors[0], label=r'\textbf{Moment-Matched Hetero-Normal}')
    leg = plt.legend(loc='center left', ncol=1, fontsize=30)

    # from https://stackoverflow.com/a/48296983
    for legobj in leg.legend_handles:
        legobj.set_linewidth(3.0)


def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*/*.npy', recursive=True)


eps = 1e-5

for env in domains_to_envs:
    width  = 6
    height = 1 + len(domains_to_envs[env]) // width
    fig, axes = plt.subplots(height, width, figsize=(22, .5 + height * 3.7))
    for position, name in enumerate(domains_to_envs[env]):
        x, y = int(position / width), position % width

        if height == 1:
            current_axis = axes[y]
        else:
            current_axis = axes[x, y]

        for pl in path_lists:
            # from https://stackoverflow.com/a/33159707
            files = listfiles(pl, name)
            files = listfiles(os.path.dirname(files[0]), '')
            files.sort(key=lambda f: int(re.sub('\D', '', f)))

            f      = files[50]
            colors = sns.color_palette('colorblind', n_colors=4)
            aux    = np.load(f, allow_pickle=True)
            target = aux.item().get('target_Q').flatten()
            mean   = aux.item().get('online_Q').flatten()
            std    = aux.item().get('online_std').flatten()


            def plot_nll(ax):
                files = listfiles(pl, name)
                set_of_seeds = list((set([os.path.dirname(f) for f in files])))

                hetero_data_array = []
                homo_data_array   = []

                for file in set_of_seeds:
                    files = listfiles(file, '')
                    files.sort(key=lambda f: int(re.sub('\D', '', f)))
                    files = files[1:]

                    hetero_data = []
                    homo_data   = []
                    for f in files:
                        rolling_aux    = np.load(f, allow_pickle=True)
                        rolling_target = rolling_aux.item().get('target_Q').flatten()
                        rolling_mean   = rolling_aux.item().get('online_Q').flatten()
                        rolling_std    = rolling_aux.item().get('online_std').flatten()

                        hetero_datapoint = (rolling_target - rolling_mean) / rolling_std
                        homo_datapoint   = (rolling_target - rolling_mean) / rolling_std.mean()

                        hetero_data.append(hetero_datapoint)
                        homo_data.append(homo_datapoint)

                    hetero_data_array.append(hetero_data)
                    homo_data_array.append(homo_data)

                hetero_data            = np.array(hetero_data_array)
                hetero_mean            = hetero_data.mean(-1, keepdims=True)
                hetero_std             = hetero_data.std(-1, keepdims=True) + eps
                hetero_logistic_spread = hetero_std * np.sqrt(3) / np.pi

                logistic_nll      = - logistic.logpdf(hetero_data, loc=hetero_mean, scale=hetero_logistic_spread).mean(axis=-1)
                logistic_nll_mean = logistic_nll.mean(0).squeeze()
                logistic_nll_std  = logistic_nll.std(0).squeeze()
                lower_lnnl        = logistic_nll_mean - logistic_nll_std
                upper_lnnl        = logistic_nll_mean + logistic_nll_std

                normal_nll      = np.mean(np.log(hetero_std * np.sqrt(2 * np.pi)) + .5 * ((hetero_data - hetero_mean) / hetero_std) ** 2, axis=-1)
                normal_nll_mean = normal_nll.mean(0).squeeze()
                normal_nll_std  = normal_nll.std(0).squeeze()
                lower_nnnl      = normal_nll_mean - normal_nll_std
                upper_nnnl      = normal_nll_mean + normal_nll_std

                homo_data   = np.array(homo_data_array)
                homo_mean   = homo_data.mean(-1, keepdims=True)
                homo_std    = homo_data.std(-1, keepdims=True) + eps

                homo_nll      = np.mean(np.log(homo_std * np.sqrt(2 * np.pi)) + .5 * ((homo_data - homo_mean) / homo_std) ** 2, axis=-1)
                homo_nll_mean = homo_nll.mean(0).squeeze()
                homo_nll_std  = homo_nll.std(0).squeeze()
                lower_hnnl    = homo_nll_mean - homo_nll_std
                upper_hnnl    = homo_nll_mean + homo_nll_std

                index = ((1 + np.arange(99)) / 100).tolist()
                phi   = (np.sqrt(5) - 1) / 2
                alpha = phi / 2

                ax.plot(index, logistic_nll_mean, color='black', linewidth=2)
                ax.fill_between(index, y1=lower_lnnl, y2=upper_lnnl, color='black', alpha=alpha)

                ax.plot(index, normal_nll_mean, color=colors[0], linewidth=2)
                ax.fill_between(index, y1=lower_nnnl, y2=upper_nnnl, color=colors[0], alpha=alpha)

                ax.plot(index, homo_nll_mean, color=colors[1], linewidth=1.2)
                ax.fill_between(index, y1=lower_hnnl, y2=upper_hnnl, color=colors[1], alpha=alpha)


            plot_nll(current_axis)
            utils._decorate_axis(current_axis, landscape=False)
            current_axis.set_title(name.replace('metaworld_', ''), fontsize=fontsize)

    if env == 'Box2D':
        # from https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
        gs = axes[2].get_gridspec()
        axes[2].remove()
        axes[3].remove()
        axes[4].remove()
        axes[5].remove()
        bigaxes = fig.add_subplot(gs[2:5])
        make_legend(bigaxes)

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
    plt.ylabel('NLL\n', fontsize=30)
    plt.tight_layout()
    utils.save_fig(plt, f'figs6/{env}')

plt.show()
