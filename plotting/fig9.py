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
    'Box2D'    : ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3']
}

path_lists = [
    # benchmark
    '../../data/camera-ready/continuous/DoubleGum/c-0.5',
    '../../data/camera-ready/continuous/DoubleGum/c-0.1',
    '../../data/camera-ready/continuous/DoubleGum/c0.0',
    '../../data/camera-ready/continuous/DoubleGum/c0.1',
    '../../data/camera-ready/continuous/DoubleGum/c0.5',
]

pretty_display = {
    r'$c=-0.5$'                   : '../../data/camera-ready/continuous/DoubleGum/c-0.5',
    r'\textbf{$c=-0.1$ (Default)}': '../../data/camera-ready/continuous/DoubleGum/c-0.1',
    r'$c=0$'                      : '../../data/camera-ready/continuous/DoubleGum/c0.0',
    r'$c=0.1$'                    : '../../data/camera-ready/continuous/DoubleGum/c0.1',
    r'$c=0.5$'                    : '../../data/camera-ready/continuous/DoubleGum/c0.5',
}

algorithms = list(pretty_display.keys())


def plot_sample_efficiency(ax, frames, point_estimates, interval_estimates, algorithms=None):
    color_palette   = sns.color_palette('colorblind', n_colors=len(algorithms))
    colors          = dict(zip(algorithms, color_palette))

    # https://stackoverflow.com/a/29498853
    for algorithm in algorithms[1:] + algorithms[:1]:
        metric_values = point_estimates[algorithm]
        lower, upper  = interval_estimates[algorithm]
        ax.plot(
            frames,
            metric_values,
            color=colors[algorithm],
            linewidth=2 if algorithm == r'\textbf{$c=-0.1$ (Default)}' else 1.2,
            label=algorithm,
        )
        phi = (np.sqrt(5) - 1) / 2
        alpha = phi / len(path_lists)
        ax.fill_between(frames, y1=lower, y2=upper, color=colors[algorithm], alpha=alpha)
    return utils._decorate_axis(ax)



def make_legend(ax, algorithms=None):
    color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
    colors        = dict(zip(algorithms, color_palette))
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for algorithm in algorithms:
        plt.plot(0, 0, color=colors[algorithm], label=algorithm)
    ax.legend(loc='center', ncol=len(algorithms))
    return ax


width = 4
height = 1
# fig, ax = plt.subplots(figsize=(7, 4.5))
# fig, axes = plt.subplots(height, width, figsize=(22, 3.5))
# fig, axes = plt.subplots(height, width, figsize=(22, 4.5))
fig, axes = plt.subplots(height, width, figsize=(22, 5))

for position, domain in enumerate(domains_to_envs):
    score_dict = {}
    for path in path_lists:
        aggregate_results = []
        for env in domains_to_envs[domain]:
            max_score = upper_bounds[domain](env)
            min_score = lower_bounds[env]
            results   = utils.load_path_name(path, env)
            results   = (results - min_score)  / (max_score - min_score)
            # print([len(r) for r in results], env)
            aggregate_results.append(results)

        shortest = min([min(len(a) for a in ar) for ar in aggregate_results])
        # print(shortest)
        cleaned_aggregate_results = []
        for ar in aggregate_results:
            seed_results = []
            for a in ar:
                seed_results.append(a[:shortest])
            cleaned_aggregate_results.append(seed_results)

        aggregate_results = np.array(cleaned_aggregate_results)
        # print(domain)
        # print(path)
        aggregate_results = aggregate_results.swapaxes(0, 1)
        # aggregate_results = aggregate_results[:, :, -1]

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

# From https://stackoverflow.com/a/24229589
color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
colors = dict(zip(algorithms, color_palette))
for algorithm in algorithms:
    plt.plot(0, 0, color=colors[algorithm], label=algorithm)
leg = plt.legend(loc='upper center', ncol=len(algorithms), fontsize=30, bbox_to_anchor=(0.5, -0.25))

# from https://stackoverflow.com/a/48296983
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)

plt.tight_layout()
utils.save_fig(plt, 'fig9')
plt.show()
