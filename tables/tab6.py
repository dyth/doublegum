import glob
import numpy as np


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


for name in envs:
    names      = []
    row_means  = []
    row_stdevs = []
    for path in path_lists:
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
            stdevs  = results.std(0)
            results = results[cut:-cut]
            means   = results.mean(0)

            names.append(name)
            row_means.append(means[-1])
            row_stdevs.append(stdevs[-1])

        except:
            print(path, name)

    biggest = max(row_means)
    row = f"\\texttt{{{name}}} "
    for rm, rs in zip(row_means, row_stdevs):
        if rm == biggest:
            row += f"& \\textbf{{{rm:.4g} $\pm$ {rs:.4g}}} "
        else:
            row += f"& {rm:.4g} $\pm$ {rs:.4g} "
    row += "\\\ "
    print(row)

