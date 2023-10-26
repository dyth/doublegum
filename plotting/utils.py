import numpy as np
import pathlib
import glob
from matplotlib.ticker import MaxNLocator


def _decorate_axis(ax, landscape=True):
    # from https://github.com/google-research/rliable/blob/46f250777f69313f813026f9d6e1cc9d4b298e2d/rliable/plot_utils.py#L70
    """Helper function for decorating plots."""
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(True, alpha=0.2)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    if landscape:
        phi = (1 + np.sqrt(5)) / 2
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    else:
        # phi = 8.5 / 11.
        phi = 0.9
        ax.set_xticks([0., .5, 1.])

    ax.set_aspect(1. / (phi * ax.get_data_ratio()), adjustable='box')
    return ax


def save_fig(fig, name):
    folder    = 'plotting/plots/'
    file_name = f'{folder}/{name}.pdf'
    parent = pathlib.Path(file_name).parent.absolute()
    pathlib.Path(parent).mkdir(parents=True, exist_ok=True)
    fig.savefig(file_name, format='pdf', bbox_inches='tight')
    print(file_name)


def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*.txt', recursive=True)


def load_path_name(path, name):
    '''for continuous control'''
    paths = listfiles(path, name)
    results = []
    for files in paths:
        result_file = np.loadtxt(files)
        result      = np.array(result_file)
        results.append(result.tolist())

    try:
        new_results = []
        min_length = min([len(r) for r in results])
        for r, f in zip(results, files):
            new_results.append(r)
        new_results = [r[:min_length] for r in new_results]
        results     = np.array(new_results)

    except:
        print(path, name)

    return results


domains_to_envs = {
    'DMC'      : ['acrobot-swingup', 'reacher-hard', 'finger-turn_hard', 'hopper-hop', 'fish-swim', 'cheetah-run', 'walker-run', 'quadruped-run', 'swimmer-swimmer15', 'humanoid-run', 'dog-run'],
    'MuJoCo'   : ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4'],
    'MetaWorld': ['metaworld_button-press-v2', 'metaworld_door-open-v2', 'metaworld_drawer-close-v2', 'metaworld_drawer-open-v2', 'metaworld_peg-insert-side-v2', 'metaworld_pick-place-v2', 'metaworld_push-v2', 'metaworld_reach-v2', 'metaworld_window-open-v2', 'metaworld_window-close-v2', 'metaworld_basketball-v2', 'metaworld_dial-turn-v2', 'metaworld_sweep-into-v2', 'metaworld_hammer-v2', 'metaworld_assembly-v2'],
    'Box2D'    : ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3'],
    'Discrete' : ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
}


def suite(env):
    for domain in domains_to_envs:
        if env in domains_to_envs[domain]:
            return domain
