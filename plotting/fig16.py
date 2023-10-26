"""
Lovingly pastiched from https://colab.research.google.com/drive/1a0pSD-1tWhMmeJeeoyZM1A-HCW3yf1xR
"""
import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

# latex text rendering, from https://stackoverflow.com/a/8384685
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']


from lower_bounds import lower_bounds
from upper_bounds import upper_bounds
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


def plot_sample_efficiency(ax, frames, point_estimates, interval_estimates, algorithms=None):
    color_palette   = sns.color_palette('colorblind', n_colors=len(algorithms))
    base_algorithms = [r'\textbf{DoubleGum, $c=-0.1$ (Ours)}', 'MoG-DDPG', 'DDPG', 'SAC', 'XQL', 'QR-DDPG', 'FinerTD3']
    colors          = dict(zip(base_algorithms, color_palette))

    # https://stackoverflow.com/a/29498853
    for algorithm in algorithms[1:] + algorithms[:1]:
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

        metric_values = point_estimates[algorithm]
        lower, upper  = interval_estimates[algorithm]
        ax.plot(
            frames,
            metric_values,
            color=colors[alg_to_base[algorithm]],
            linewidth=2 if algorithm == r'\textbf{DoubleGum, $c=-0.1$ (Ours)}' else 1.2,
            label=algorithm,
            linestyle=linestyle,
        )
        phi = (np.sqrt(5) - 1) / 2
        alpha = phi / len(path_lists)
        ax.fill_between(frames, y1=lower, y2=upper, color=colors[alg_to_base[algorithm]], alpha=alpha)
    return utils._decorate_axis(ax)


width = 4
height = 1
# fig, ax = plt.subplots(figsize=(7, 4.5))
# fig, axes = plt.subplots(height, width, figsize=(22, 3.5))
# fig, axes = plt.subplots(height, width, figsize=(22, 4.5))
fig, axes = plt.subplots(height, width, figsize=(22, 5.5))

for position, domain in enumerate(domains_to_envs):
    score_dict = {}
    for path in path_lists:
        aggregate_results = []
        for env in domains_to_envs[domain]:
            max_score = upper_bounds[domain](env)
            min_score = lower_bounds[env]
            results   = utils.load_path_name(path, env)
            results   = (results - min_score)  / (max_score - min_score)
            aggregate_results.append(results)

        shortest                  = min([min(len(a) for a in ar) for ar in aggregate_results])
        # print(shortest, len(aggregate_results))
        cleaned_aggregate_results = []
        for ar in aggregate_results:
            seed_results = []
            for a in ar:
                seed_results.append(a[:shortest])
            cleaned_aggregate_results.append(seed_results)

        aggregate_results = np.array(cleaned_aggregate_results)
        aggregate_results = aggregate_results.swapaxes(0, 1)
        score_dict[path] = aggregate_results

    for pd in pretty_display:
        score_dict[pd] = score_dict.pop(pretty_display[pd])


    frames = np.arange(100)
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm, reps=2000)
    # iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm, reps=20)

    x, y = int(position / width), position % width

    plot_sample_efficiency(axes[y], frames/100, iqm_scores, iqm_cis, algorithms=algorithms)
    axes[y].set_title(domain, fontsize=30)


# From https://stackoverflow.com/a/53172335
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Timesteps (in millions)', fontsize=30)
plt.ylabel('IQM', fontsize=30)


color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
base_algorithms = [r'\textbf{DoubleGum, $c=-0.1$ (Ours)}', 'MoG-DDPG', 'DDPG', 'SAC', 'XQL', 'QR-DDPG', 'FinerTD3']
colors = dict(zip(base_algorithms, color_palette))
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
leg = plt.legend(loc='upper center', ncol=4, fontsize=30, bbox_to_anchor=(0.5, -0.25))
# leg = plt.legend(loc='upper center', ncol=3, fontsize=30, bbox_to_anchor=(0.5, -.25))

# from https://stackoverflow.com/a/48296983
for legobj in leg.legend_handles:
    legobj.set_linewidth(3.0)

plt.tight_layout()
utils.save_fig(plt, 'fig16')
plt.show()
