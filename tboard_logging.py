import subprocess
from tensorboardX import SummaryWriter


def setup_tensorboard(folder, args):
    writer = SummaryWriter(folder)
    # from https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/sac_continuous_action.py
    names = [f'|{key}|{value}|' for key, value in vars(args).items()]
    writer.add_text('hyperparameters', '|param|value|\n|-|-|\n%s' % ('\n'.join(names)))

    # from https://github.com/mila-iqia/babyai/blob/master/scripts/train_rl.py#L157
    try:
        last_commit = subprocess.check_output('git log -n1', shell=True).decode('utf-8')
        writer.add_text('last commit', last_commit)
    except subprocess.CalledProcessError:
        writer.add_text('last commit', 'Could not figure out the last commit')

    try:
        diff = subprocess.check_output('git diff', shell=True).decode('utf-8')
        if diff:
            writer.add_text('git diff', f"```{diff}```")
    except subprocess.CalledProcessError:
        writer.add_text('last diff', 'Could not figure out the last diff')

    return writer


def log_to_tensorboard(writer, info, timestep, folder='values', name=''):
    for i in info:
        value = info[i].squeeze()
        if value is not None:
            if hasattr(value, "__len__") and len(value.shape) > 0:
                sub_info = dict(zip(range(len(value)), value))
                log_to_tensorboard(writer, sub_info, timestep, folder=folder, name=f'{name}{i}_')
            else:
                writer.add_scalar(f'{folder}/{name}{i}', value, timestep)
