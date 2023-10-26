"""
Lovingly pastiched from https://colab.research.google.com/drive/1a0pSD-1tWhMmeJeeoyZM1A-HCW3yf1xR
"""
import numpy as np
from rliable import library as rly
from rliable import metrics

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


def plot_sample_efficiency(point_estimates, algorithms=None):
    print(' & '.join(algorithms) + " \\\ ")

    row_means = []
    for algorithm in algorithms:
        row_means.append(point_estimates[algorithm][-1])

    biggest = max(row_means)
    row = f"All "
    for rm in row_means:
        if rm == biggest:
            row += f"& \\textbf{{{rm:.4g}}} "
        else:
            row += f"& {rm:.4g} "
    row += "\\\ "

    print(row)


score_dict = {}
for path in path_lists:
    aggregate_results = []
    for position, domain in enumerate(domains_to_envs):
        for env in domains_to_envs[domain]:
            max_score = upper_bounds[domain](env)
            min_score = lower_bounds[env]
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

    aggregate_results = np.array(cleaned_aggregate_results)
    aggregate_results = aggregate_results.swapaxes(0, 1)
    score_dict[path] = aggregate_results

for pd in pretty_display:
    score_dict[pd] = score_dict.pop(pretty_display[pd])


iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
# iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm, reps=2000)
iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm, reps=20)

plot_sample_efficiency(iqm_scores, algorithms=algorithms)
