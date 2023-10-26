'''
Plotting MuJoCo benchmark tensorboard scalars in matplotilb
Adapted from https://www.tensorflow.org/tensorboard/dataframe_api
'''
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
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
import utils

rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']


envs = {
    'acrobot-swingup': 'DMC',
    # 'reacher-hard': '(obs 6, act 2)',
    # 'finger-turn_hard': '(obs 12, act 2)',
    # 'hopper-hop': '(obs 15, act 4)',
    # 'fish-swim': '(obs 24, act 5)',
    # 'cheetah-run': '(obs 17, act 6)',
    # 'walker-run': '(obs 24, act 6)',
    # 'quadruped-run': '(obs 58, act 12)',
    # 'swimmer-swimmer15': '(obs 61, act 14)',
    # 'humanoid-run': '(obs 67, act 21)',
    # 'dog-run': '(obs 227, act 38)',

    'Hopper-v4': 'MuJoCo',
    # 'HalfCheetah-v4': '(obs 17, act 6)',
    # 'Walker2d-v4': '(obs 17, act 6)',
    # 'Ant-v4': 'MuJoCo',
    # 'Humanoid-v4': '(obs 376, act 17)',

    'BipedalWalker-v3': 'Box2d',
    # 'BipedalWalkerHardCore-v3': '(obs 24, act 4)',

    'metaworld_button-press-v2'   : 'MetaWorld',
    # 'metaworld_door-open-v2'      : '',
    # 'metaworld_drawer-close-v2'   : '',
    # 'metaworld_drawer-open-v2'    : '',
    # 'metaworld_hammer-v2'         : '',
    # 'metaworld_peg-insert-side-v2': '',
    # 'metaworld_pick-place-v2'     : '',
    # 'metaworld_push-v2'           : '',
    # 'metaworld_reach-v2'          : '',
    # 'metaworld_window-open-v2'    : '',
    # 'metaworld_window-close-v2'   : '',
    # 'metaworld_basketball-v2'     : '',
    # 'metaworld_dial-turn-v2'      : '',
    # 'metaworld_sweep-into-v2'     : '',
    # 'metaworld_assembly-v2'       : ''
}



jaxrl_path_lists = []

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


height = 1
width  = 4
fig, axes = plt.subplots(height, width, figsize=(22, 5))
# fig, axes = plt.subplots(height, width)
for position, name in enumerate(envs):
    colors = sns.color_palette('colorblind', n_colors=5)
    for path, color in zip(path_lists, colors):
        paths = listfiles(path, name)
        results = []
        for files in paths:
            print(files)
            tensorfile = load_tf(files)
            # result = tensorfile['estimation/overestimation']
            result = tensorfile['critic/target_Q_mean']
            results.append(result.tolist())

        try:
            new_results = []
            min_length = min([len(r) for r in results])
            for r, f in zip(results, files):
                new_results.append(r)
            new_results = [r[:min_length] for r in new_results]

            results         = np.array(new_results)
            proportiontocut = 0.25
            samples         = 12
            cut             = int(proportiontocut * samples)

            results.sort(0)
            stdev = results.std(0)
            results = results[cut:-cut]

            means = results.mean(0)
            # index = (10000 * np.arange(results.shape[1])).tolist()
            index = np.arange(results.shape[1]).tolist()


            x, y = int(position/width), position%width
            axes[y].plot(index, means, color=color)
            axes[y].set_title(f"{envs[name]}: {name.replace('metaworld_', '')}", fontsize=30)
            phi = (np.sqrt(5) - 1) / 2
            alpha = phi / len(path_lists)
            axes[y].fill_between(index, np.array(means)-stdev, np.array(means)+stdev, alpha=alpha, color=color)
            # axes[y].set_yscale('symlog')
            utils._decorate_axis(axes[y])

        except:
            print(path, name)


# From https://stackoverflow.com/a/53172335
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Timesteps (in millions)', fontsize=30)
plt.ylabel('Magnitude', fontsize=30)

algorithms = [r'c=-0.5', r'c=-0.1', r'c=0.0', r'c=0.1', r'c=0.5']
color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
colors = dict(zip(algorithms, color_palette))
for algorithm in algorithms:
    plt.plot(0, 0, color=colors[algorithm], label=algorithm)
leg = plt.legend(loc='upper center', ncol=len(algorithms), fontsize=30, bbox_to_anchor=(0.5, -0.25))


# from https://stackoverflow.com/a/48296983
for legobj in leg.legend_handles:
    legobj.set_linewidth(3.0)

plt.tight_layout()
utils.save_fig(plt, 'fig2')
plt.show()
