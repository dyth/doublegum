'''
Plotting MuJoCo benchmark tensorboard scalars in matplotilb
Adapted from https://www.tensorflow.org/tensorboard/dataframe_api
'''
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
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
    '../../data/camera-ready/continuous/DoubleGum/c-0.5',
    '../../data/camera-ready/continuous/DoubleGum/c-0.1',
    '../../data/camera-ready/continuous/DoubleGum/c0.0',
    '../../data/camera-ready/continuous/DoubleGum/c0.1',
    '../../data/camera-ready/continuous/DoubleGum/c0.5',
]


'''
Index(['returns/train', 'estimation/observed', 'estimation/episode_len',
       'estimation/online_expected', 'estimation/target_expected',
       'returns/test', 'values/fps', 'correlation/sq_std',
       'correlation/sq_var', 'correlation/td_std', 'correlation/td_var',
       'p_values/sq_std', 'p_values/sq_var', 'p_values/td_std',
       'p_values/td_var', 'critic/critic_loss', 'critic/critic_loss_std',
       'critic/log_std_mean', 'critic/log_std_std', 'critic/neg_td_mean',
       'critic/neg_weight_mean', 'critic/online_Q', 'critic/online_Q_std',
       'critic/online_std_mean', 'critic/online_std_std', 'critic/pos_td_mean',
       'critic/pos_weight_mean', 'critic/target_Q_mean', 'critic/target_Q_std',
       'critic/td_increase', 'critic/td_loss', 'critic/td_loss_std',
       'critic/td_ratio', 'critic/weight_increase', 'critic/weight_ratio',
       'critic/weighted_mse_mean', 'critic/weighted_mse_std',
       'actor/action_diff', 'actor/action_diff_l1', 'actor/actor_loss_mean',
       'actor/actor_loss_std']
'''




def load_tf(dirname):
    '''
    https://gist.github.com/willwhitney/9cecd56324183ef93c2424c9aa7a31b4
    '''
    ea = event_accumulator.EventAccumulator(dirname, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    dframes = {}
    # mnames = ea.Tags()['scalars']
    # mnames = ['estimation/observed', 'estimation/target_expected',]
    # mnames = ['estimation/overestimation',]
    mnames = ['critic/target_Q_mean']

    for n in mnames:
        dframe = pd.DataFrame(ea.Scalars(n), columns=["wall_time", "step", 'value'])
        dframe.drop("wall_time", axis=1, inplace=True)
        dframe = dframe.set_index("step")
        dframe = dframe.rename(columns={'value': n})
        dframes[n] = dframe
    return pd.concat([v for k,v in dframes.items()], axis=1)


def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*/*tfevents*', recursive=True)


# algorithms = list(pretty_display.keys())
algorithms = [r'$c=-0.5$', r'\textbf{$c=-0.1$ (Default)}', r'$c=0.0$', r'$c=0.1$', r'$c=0.5$']

def make_legend(ax, algorithms=None):
    color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
    colors        = dict(zip(algorithms, color_palette))
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for algorithm in algorithms:
        ax.plot(0, 0, color=colors[algorithm], label=algorithm)
    leg = ax.legend(loc='center left', fontsize=fontsize)

    # from https://stackoverflow.com/a/48296983
    for legobj in leg.legend_handles:
        legobj.set_linewidth(3.0)


for env in domains_to_envs:
    width  = 6
    height = 1 + len(domains_to_envs[env]) // width
    fig, axes = plt.subplots(height, width, figsize=(22, .5 + height * 4))
    for position, name in enumerate(domains_to_envs[env]):
        x, y = int(position / width), position % width

        colors = sns.color_palette('colorblind', n_colors=len(algorithms))

        rot_path_lists = path_lists[1:] + path_lists[:1]
        rot_colors     = colors[1:] + colors[:1]
        for path, color in zip(rot_path_lists, rot_colors):
            paths = listfiles(path, name)
            results = []

            for files in paths:
                print(files)
                tensorfile = load_tf(files)
                # result = tensorfile['estimation/overestimation']
                result = tensorfile['critic/target_Q_mean']
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
                axes[y].plot(index, means, color=color, linewidth=2 if 'c-0.1' in path else 1.2)
                axes[y].set_title(name.replace('metaworld_', ''), fontsize=fontsize)
                axes[y].fill_between(index, np.array(means)-stdev, np.array(means)+stdev, alpha=alpha, color=color)
            else:
                axes[x, y].plot(index, means, color=color, linewidth=2 if 'c-0.1' in path else 1.2)
                axes[x, y].set_title(name.replace('metaworld_', ''), fontsize=fontsize)
                axes[x, y].fill_between(index, np.array(means)-stdev, np.array(means)+stdev, alpha=alpha, color=color)

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
            # make_legend(axes[x, y+1], algorithms=algorithms)
            for i in range(y+1, width):
                axes[x, i].remove()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Timesteps (in millions)', fontsize=30)
    plt.ylabel('Magnitude\n', fontsize=30)
    plt.tight_layout()
    utils.save_fig(plt, f'figs7/{env}')

plt.show()
