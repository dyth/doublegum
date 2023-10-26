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
    'CartPole-v1': '(obs 6, act 1)',
    'Acrobot-v1': '(obs 12, act 2)',
    'MountainCar-v0': '(obs 17, act 6)',
}


path_lists = [
    '../../data/camera-ready/discrete/DoubleGum',
    '../../data/camera-ready/discrete/DQN',
    '../../data/camera-ready/discrete/DuelingDDQN',
]


def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*.txt', recursive=True)


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



height = 1
width  = 3
fig, axes = plt.subplots(height, width, figsize=(22, 6))
for position, name in enumerate(envs):
    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # colors = sns.color_palette('colorblind', n_colors=len(path_lists))
    color_palette = sns.color_palette('colorblind', n_colors=len(path_lists))
    colors = dict(zip(path_lists, color_palette))

    for path in path_lists[1:] + path_lists[:1]:
        paths = listfiles(path, name)
        results = []
        for files in paths:
            result_file = np.loadtxt(files)
            result      = np.array(result_file)
            results.append(result.tolist())

        try:
            new_results = []
            min_length  = min([len(r) for r in results])
            for r in results:
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
            index = (np.arange(results.shape[1])).tolist()

            color = colors[path]
            x, y = int(position/width), position%width
            axes[y].plot(index, means, color=color, linewidth=2 if 'DoubleGum' in path else 1.2)
            axes[y].set_title(name, fontsize=30)

            phi = (np.sqrt(5) - 1) / 2
            alpha = phi / len(path_lists)
            axes[y].fill_between(index, np.array(means)-stdev, np.array(means)+stdev, alpha=alpha, color=color)
            utils._decorate_axis(axes[y])

        except:
            print(path, name)


# From https://stackoverflow.com/a/53172335
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Timesteps (in thousands)', fontsize=30)
plt.ylabel('IQM of Return', fontsize=30)

# From https://stackoverflow.com/a/24229589
algorithms = [r'\textbf{DoubleGum (Ours)}', 'DQN', 'Dueling DDQN']
color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
colors = dict(zip(algorithms, color_palette))
for algorithm in algorithms:
    plt.plot(0, 0, color=colors[algorithm], label=algorithm)
leg = plt.legend(loc='upper center', ncol=len(algorithms), fontsize=30, bbox_to_anchor=(0.5, -0.25))

# from https://stackoverflow.com/a/48296983
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)

plt.tight_layout()
utils.save_fig(plt, 'fig3')
plt.show()
