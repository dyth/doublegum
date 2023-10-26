import re
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams.update({'font.size': 20})
# latex text rendering, from https://stackoverflow.com/a/8384685
from matplotlib import rc
import utils
from scipy.stats import logistic

rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']


path_lists = [
    f'../../data/camera-ready/continuous/DoubleGum/c-0.1/td',
]


envs = [
    'acrobot-swingup', 'reacher-hard', 'finger-turn_hard', 'hopper-hop', 'fish-swim', 'cheetah-run', 'walker-run',
    'quadruped-run', 'swimmer-swimmer15', 'humanoid-run', 'dog-run',

    'Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4',

    'metaworld_button-press-v2', 'metaworld_door-open-v2', 'metaworld_drawer-close-v2',
    'metaworld_drawer-open-v2', 'metaworld_peg-insert-side-v2', 'metaworld_pick-place-v2',
    'metaworld_push-v2', 'metaworld_reach-v2', 'metaworld_window-open-v2', 'metaworld_window-close-v2',
    'metaworld_basketball-v2', 'metaworld_dial-turn-v2', 'metaworld_sweep-into-v2', 'metaworld_hammer-v2',
    'metaworld_assembly-v2',

    'BipedalWalker-v3', 'BipedalWalkerHardcore-v3',
]


def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*/*.npy', recursive=True)



def make_legend(ax, algorithms=None):
    color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
    colors        = dict(zip(algorithms, color_palette))
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for algorithm in algorithms:
        plt.plot(0, 0, color=colors[algorithm], label=algorithm)
    # ax.legend(loc='center', ncol=len(algorithms), fontsize=30)
    ax.legend(loc='center', fontsize=30)
    return ax


eps = 1e-5
for env in envs:
    height, width = 1, 3
    # fig, axes = plt.subplots(height, width, figsize=(22, 3.5), width_ratios=[1, 1, 1, .5])
    # fig, axes = plt.subplots(height, width, figsize=(22, 5), gridspec_kw = {'width_ratios': [1, 1, 1, .3]})
    fig, axes = plt.subplots(height, width, figsize=(22, 5))

    for pl in path_lists:
        # from https://stackoverflow.com/a/33159707
        files = listfiles(pl, env)
        files = listfiles(os.path.dirname(files[0]), '')
        files.sort(key=lambda f: int(re.sub('\D', '', f)))

        # f = files[-1]
        f = files[50]
        colors = sns.color_palette('colorblind', n_colors=4)

        aux        = np.load(f, allow_pickle=True)
        target     = aux.item().get('target_Q').flatten()
        mean       = aux.item().get('online_Q').flatten()
        std        = aux.item().get('online_std').flatten()


        def plot_homoscedastic(data, index):
            emp_mean            = data.mean()
            emp_std             = data.std()
            emp_spread          = emp_std * np.sqrt(6) / np.pi
            emp_loc             = emp_mean + np.euler_gamma * emp_spread

            spread = 3
            low    = emp_mean - spread*emp_std
            high   = emp_mean + spread*emp_std
            axes[index].set_xlim(low, high)

            bins = int((data.max() - data.min()) / (high - low) * 100)
            bins = axes[index].hist(data, bins, density=True, color='silver', histtype='barstacked')[1]

            normal_pdf = 1 / (emp_std * np.sqrt(2 * np.pi)) * np.exp(-.5 * ((bins - emp_mean) / emp_std) ** 2)
            axes[index].plot(bins, normal_pdf, color=colors[1], linewidth=1.8)

            z = (bins - emp_loc) / emp_spread
            gumbel_pdf = (1 / emp_spread) * np.exp(-z - np.exp(-z))
            axes[index].plot(bins, gumbel_pdf, color=colors[2], linewidth=1.8)
            utils._decorate_axis(axes[index])


        def plot_heteroscedastic(data, index):
            emp_mean            = data.mean()
            emp_std             = data.std()
            emp_logistic_spread = emp_std * np.sqrt(3) / np.pi

            spread = 3
            low    = emp_mean - spread*emp_std
            high   = emp_mean + spread*emp_std
            axes[index].set_xlim(low, high)

            bins = int((data.max() - data.min()) / (high - low) * 100)
            bins = axes[index].hist(data, bins, density=True, color='silver', histtype='barstacked')[1]

            normal_pdf = 1 / (emp_std * np.sqrt(2 * np.pi)) * np.exp(-.5 * ((bins - emp_mean) / emp_std) ** 2)
            axes[index].plot(bins, normal_pdf, color=colors[0], linewidth=3)

            z            = (bins - emp_mean) / emp_logistic_spread
            logistic_pdf = np.exp(- z) / (emp_logistic_spread * (1 + np.exp(-z)) ** 2)
            axes[index].plot(bins, logistic_pdf, color='black', linewidth=3)
            utils._decorate_axis(axes[index])


        def plot_nll(ax):
            files = listfiles(pl, env)
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
                    rolling_aux = np.load(f, allow_pickle=True)
                    # print(aux.item().keys()) # 'online_Q', 'online_std', 'target_Q', 'target_std'

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
            homo_spread = homo_std * np.sqrt(6) / np.pi
            homo_loc    = homo_mean - np.euler_gamma * homo_spread

            # z               = (homo_data - homo_loc) / homo_spread
            # gumbel_nll      = np.mean(np.log(homo_spread) + z + np.exp(-z), axis=-1)
            # gumbel_nll      = gumbel.logpdf(hetero_data, hetero_mean, hetero_logistic_spread).mean(axis=-1)
            # gumbel_nll_mean = gumbel_nll.mean(0).squeeze()
            # gumbel_nll_std  = gumbel_nll.std(0).squeeze() + eps
            # lower_gnnl      = gumbel_nll_mean - gumbel_nll_std
            # upper_gnnl      = gumbel_nll_mean + gumbel_nll_std

            homo_nll      = np.mean(np.log(homo_std * np.sqrt(2 * np.pi)) + .5 * ((homo_data - homo_mean) / homo_std) ** 2, axis=-1)
            homo_nll_mean = homo_nll.mean(0).squeeze()
            homo_nll_std  = homo_nll.std(0).squeeze()
            lower_hnnl    = homo_nll_mean - homo_nll_std
            upper_hnnl    = homo_nll_mean + homo_nll_std

            index = ((1 + np.arange(99)) / 100).tolist()
            phi   = (np.sqrt(5) - 1) / 2
            alpha = phi / 2

            ax.plot(index, logistic_nll_mean, color='black', linewidth=3)
            ax.fill_between(index, y1=lower_lnnl, y2=upper_lnnl, color='black', alpha=alpha)

            ax.plot(index, normal_nll_mean, color=colors[0], linewidth=3)
            ax.fill_between(index, y1=lower_nnnl, y2=upper_nnnl, color=colors[0], alpha=alpha)

            # ax.plot(index, gumbel_nll_mean, color=colors[2], linewidth=3)
            # ax.fill_between(index, y1=lower_gnnl, y2=upper_gnnl, color=colors[2], alpha=alpha)

            ax.plot(index, homo_nll_mean, color=colors[1], linewidth=1.8)
            ax.fill_between(index, y1=lower_hnnl, y2=upper_hnnl, color=colors[1], alpha=alpha)

            utils._decorate_axis(ax)


        index = 0
        plot_homoscedastic((target - mean) / std.mean(), index)
        axes[index].set_xlabel('TD Error', fontsize=30)
        axes[index].set_ylabel('Frequency Density', fontsize=30)
        # from https://stackoverflow.com/a/18346779
        axes[index].text(0.05, 1.06, 'a', transform=axes[index].transAxes, fontsize=40, va='top')

        index = 1
        plot_heteroscedastic((target - mean) / std, index)
        axes[index].set_xlabel('Standardized TD Error', fontsize=30)
        axes[index].set_ylabel('Frequency Density', fontsize=30)
        axes[index].text(0.05, 1.06, 'b', transform=axes[index].transAxes, fontsize=40, va='top')

        index = 2
        plot_nll(axes[index])
        axes[index].set_xlabel('Timesteps (in millions)', fontsize=30)
        axes[index].set_ylabel('NLL', fontsize=30)
        axes[index].text(0.05, 1.06, 'c', transform=axes[index].transAxes, fontsize=40, va='top')


    plt.tight_layout()
    if env != envs[-1]:
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        if 'metaworld_' in env:
            env = env.replace('metaworld_', '')
        plt.xlabel(f'\n{env}: {utils.suite(env)}', fontsize=40)
        utils.save_fig(plt, f'fig6/{env}')
        # plt.show()


# From https://stackoverflow.com/a/24229589
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.plot(0, 0, color='silver' , label='Empirical (Histogram)')
plt.plot(0, 0, color=colors[1], label='Homo-Normal')
plt.plot(0, 0, color='black', label=r'\textbf{Hetero-Logistic (expected)}')
plt.plot(0, 0, color=colors[2], label='Homo-Gumbel (Garg et al., 2023)')
plt.plot(0, 0, color=colors[0], label=r'\textbf{Moment-Matched Hetero-Normal}')
leg = plt.legend(loc='upper center', ncol=3, fontsize=30, bbox_to_anchor=(0.5, -.25))

# from https://stackoverflow.com/a/48296983
for legobj in leg.legend_handles:
    legobj.set_linewidth(3.0)

# plt.tight_layout()
utils.save_fig(plt, f'fig6/legend')
plt.show()
