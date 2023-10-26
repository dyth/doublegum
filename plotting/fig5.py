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
    r'\textbf{DoubleGum, best $c$, (Ours)}': DoubleGum_paths,
    'MoG-DDPG (untuned)'                   : MoG_paths,
    'Best of DDPG/TD3'                     : DDPG_paths,
    'SAC (best w/wo Twin)'                 : SAC_paths,
    r'XQL (best of $\beta$ w/wo Twin)'     : XQL_paths,
    'QR-DDPG (best w/wo Twin)'             : QR_DDPG_paths,
    'FinerTD3 (best pessimism)'            : FinerTD3_paths,
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
    'Best of DDPG/TD3'                     : DDPG,
    'SAC (best w/wo Twin)'                 : SAC,
    r'XQL (best of $\beta$ w/wo Twin)'     : XQL,
    'QR-DDPG (best w/wo Twin)'             : QR_DDPG,
    'FinerTD3 (best pessimism)'            : FinerTD3,
}

algorithms = list(pretty_display.keys())


def plot_sample_efficiency(ax, frames, point_estimates, interval_estimates, algorithms=None):
    color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
    colors        = dict(zip(algorithms, color_palette))
    for algorithm in algorithms[1:] + algorithms[:1]:
        metric_values = point_estimates[algorithm]
        lower, upper  = interval_estimates[algorithm]
        ax.plot(
            frames,
            metric_values,
            color=colors[algorithm],
            linewidth=2 if 'DoubleGum' in algorithm else 1.2,
            label=algorithm
        )
        phi = (np.sqrt(5) - 1) / 2
        alpha = phi / len(algorithms)
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


width = 1
height = 1
# fig, ax = plt.subplots(figsize=(7, 4.5))
# fig, axes = plt.subplots(height, width, figsize=(13, 8.5))
fig, axes = plt.subplots(height, width, figsize=(22, 5))

score_dict = {}
for algorithm in algorithms:
    aggregate_results = []
    for position, domain in enumerate(domains_to_envs):
        for env in domains_to_envs[domain]:
            max_score = upper_bounds[domain](env)
            min_score = lower_bounds[env]
            path      = all_paths[algorithm][pretty_display[algorithm][domain]]
            results   = utils.load_path_name(path, env)
            results   = (results - min_score)  / (max_score - min_score)
            aggregate_results.append(results)

    shortest                  = min([min(len(a) for a in ar) for ar in aggregate_results])
    cleaned_aggregate_results = []
    for ar in aggregate_results:
        seed_results = []
        for a in ar:
            seed_results.append(a[:shortest])
        cleaned_aggregate_results.append(seed_results)

    aggregate_results     = np.array(cleaned_aggregate_results)
    aggregate_results     = aggregate_results.swapaxes(0, 1)
    score_dict[algorithm] = aggregate_results


frames = np.arange(100)
iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm, reps=2000)
# iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm, reps=20)

plot_sample_efficiency(axes, frames/100, iqm_scores, iqm_cis, algorithms=algorithms)
axes.set_title('Tuned per suite', fontsize=30)


# From https://stackoverflow.com/a/53172335
fig.add_subplot(111, frameon=False)
phi = (1 + np.sqrt(5)) / 2
plt.gca().set_aspect(1. / (phi * plt.gca().get_data_ratio()), adjustable='box')
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Timesteps (in millions)', fontsize=30)
plt.ylabel('IQM', fontsize=30)

# From https://stackoverflow.com/a/24229589
color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
colors = dict(zip(algorithms, color_palette))
for algorithm in algorithms:
    plt.plot(0, 0, color=colors[algorithm], label=algorithm)
leg = plt.legend(ncol=2, fontsize=30, loc='center left', bbox_to_anchor=(1, 0.5))

# from https://stackoverflow.com/a/48296983
for legobj in leg.legend_handles:
    legobj.set_linewidth(3.0)

plt.tight_layout()
utils.save_fig(plt, 'fig5')
plt.show()
