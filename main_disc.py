import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pathlib
import time
from jax.lib import xla_bridge

from cpprb import ReplayBuffer

from timer import Timer
import tboard_logging
from parser import parser_disc
from wrappers import make_env, VideoRecorder
from policies_disc.agents import DQN, DuellingDDQN, DuellingDQN, DDQN, DoubleGum


def create_paths(args):
    file_name = f'{args.env}_{args.seed}_{int(time.time())}'
    file_dir  = f'{args.policy}/{args.name}/{file_name}'

    results_path     = f'{args.folder}/results/{file_dir}'
    results_path_dir = os.path.dirname(results_path)
    pathlib.Path(results_path_dir).mkdir(parents=True, exist_ok=True)

    tboard_path     = f'{args.folder}/tensorboard/{file_dir}'
    tboard_path_dir = os.path.dirname(tboard_path)
    pathlib.Path(tboard_path_dir).mkdir(parents=True, exist_ok=True)
    writer = tboard_logging.setup_tensorboard(tboard_path, args)

    if args.log_td:
        td_path     = f'{args.folder}/td/{file_dir}'
        # td_path_dir = os.path.dirname(td_path)
        pathlib.Path(td_path).mkdir(parents=True, exist_ok=True)
    else:
        td_path = None

    if args.video:
        video_path     = f'{args.folder}/video/{file_dir}'
        video_path_dir = os.path.dirname(video_path)
        pathlib.Path(video_path_dir).mkdir(parents=True, exist_ok=True)
    else:
        video_path = None

    return results_path, writer, video_path, td_path


def create_replay_buffer(args, state_dim):
    sample_type = {
        'obs'     : {"shape": state_dim},
        'act'     : {},
        'rew'     : {},
        'next_obs': {"shape": state_dim},
        'done'    : {},
    }
    if args.nstep > 1:
        Nstep = {
            "size" : args.nstep,
            "gamma": args.discount,
            "rew"  : "rew",
            "next" : "next_obs"
        }
    else:
        Nstep = None
    replay_buffer = ReplayBuffer(int(args.replay_size), sample_type, Nstep=Nstep)
    return replay_buffer


def eval_policy(policy, eval_env, eval_episodes, save_folder=None):
    if save_folder:
        eval_env = VideoRecorder(eval_env, save_folder=save_folder)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done     = False
        while not done:
            action                        = policy.select_action(np.array(state))
            state, reward, term, trunc, _ = eval_env.step(action)
            done                          = term or trunc
            avg_reward                   += reward
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def train(args, policy, evaluations, env, eval_env, replay_buffer, episode_num,
          timer, results_path, video_path, td_path, writer):
    state, _            = env.reset()
    episode_return      = 0
    episode_disc_return = 0
    episode_len         = 0
    done_timesteps      = 0

    if args.get_random:
        eval_episodes = 10000
        avg_reward = 0.
        for _ in range(eval_episodes):
            state, _ = eval_env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                # action = policy.select_action(np.array(state))
                state, reward, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                avg_reward += reward
        avg_reward /= eval_episodes
        print(f'{args.env}: {avg_reward}')
        exit()

    # Evaluate untrained policy
    if args.jit:
        if args.video:
            save_folder = f"{video_path}/0"
        else:
            save_folder = None
        if args.evaluate and (done_timesteps == 0):
            evaluations.append(eval_policy(policy, eval_env, args.eval_episodes, save_folder=save_folder))
            np.savetxt(f"{results_path}.txt", evaluations, fmt='%.4g')

    for t in range(int(args.max_timesteps)):
        episode_len += 1

        # Select action randomly or according to policy
        if (t < args.start_timesteps) and args.random_sampling:
            action = env.action_space.sample()
        else:
            action = policy.sample_action(np.array(state))

        # Perform action
        next_state, reward, term, trunc, info = env.step(action)
        done = term or trunc

        # Store data in replay buffer
        replay_buffer.add(
            obs=state,
            act=action,
            rew=reward,
            next_obs=next_state,
            done=int(term)
        )

        episode_return      += reward
        episode_disc_return *= args.discount
        episode_disc_return += reward

        state = next_state

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            network_info = policy.train(replay_buffer, args.replay_ratio * args.batch_size)

            if (t+1) % args.log_freq == 0:
                network_boards = [
                    'grads', 'params',
                    # 'online_l2', 'target_l2', #'online_std', 'target_std', 'coadapt', 'coadapt_disc', 'diverge'
                ]
                for cb in network_boards:
                    if cb in network_info:
                        tboard_logging.log_to_tensorboard(writer, network_info.pop(cb), t, f'network_{cb}')

                if 'grads' in network_info:
                    tboard_logging.log_to_tensorboard(writer, network_info.pop('grads'), t, 'network_grads')
                if 'layers' in network_info:
                    tboard_logging.log_to_tensorboard(writer, network_info.pop('layers'), t, 'network_layers')
                if 'params' in network_info:
                    tboard_logging.log_to_tensorboard(writer, network_info.pop('params'), t, 'network_params')

                tboard_logging.log_to_tensorboard(writer, network_info, t, 'network')

        if done:
            replay_buffer.on_episode_end()
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            writer.add_scalar(f'returns/train'         , episode_return     , t)
            writer.add_scalar(f'estimation/observed'   , episode_disc_return, episode_num)
            writer.add_scalar(f'estimation/episode_len', episode_len        , episode_num)
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_len} Reward: {episode_return:.3f}")
            # Reset environment
            state, _            = env.reset()
            episode_return      = 0
            episode_len         = 0
            episode_disc_return = 0
            episode_num        += 1
            tboard_logging.log_to_tensorboard(writer, policy.online_network(state), episode_num, 'estimation')
            writer.add_scalar(f'estimation/target_expected', policy.target_network(state), episode_num)

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:

            if args.log_td:
                ds = [policy.log_td_loss(replay_buffer) for i in range(10)]
                d  = {}
                for k in ds[0].keys():
                    d[k] = np.concatenate(list(d[k] for d in ds)) # from https://stackoverflow.com/a/5946359
                np.save(f"{td_path}/{t}.npy", d)

            if args.video and ((t + 1) % args.video_freq == 0):
                save_folder = f"{video_path}/{t}"
            else:
                save_folder = None

            if args.evaluate:
                test_return = eval_policy(policy, eval_env, args.eval_episodes, save_folder=save_folder)
                writer.add_scalar(f'returns/test', test_return, t)
                evaluations.append(test_return)
                np.savetxt(f"{results_path}.txt", evaluations, fmt='%.4g')

            fps = timer.steps_per_sec(t + 1)
            writer.add_scalar(f'values/fps', fps, t)
            print(f'Step {t+1}.  Total time cost {timer.time_cost():.4g}s.  Steps per sec: {fps:.4g}')

        writer.flush()
    return evaluations, episode_num, replay_buffer


def main():
    args = parser_disc()

    env        = make_env(args.env, args.seed    , None, continuous=False)
    eval_env   = make_env(args.env, args.seed+100, None, continuous=False)

    obs         = env.observation_space.sample()
    action      = env.action_space.sample()
    obs_dim     = env.observation_space.shape[0]
    num_actions = env.action_space.n

    if args.get_stats:
        print(f"'{args.env}': '(obs {obs_dim}, act {num_actions})'")
        exit()

    results_path, writer, video_path, td_path = create_paths(args)

    print('Start training.')
    print('Backend:'   , xla_bridge.get_backend().platform)
    print('state_dim:' , obs_dim )
    print('num_actions:', num_actions)
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    policies = {
        'DoubleGum'   : DoubleGum.DoubleGum,
        'DQN'         : DQN.DQN,
        'DDQN'        : DDQN.DDQN,
        'DuellingDQN' : DuellingDQN.DuellingDQN,
        'DuellingDDQN': DuellingDDQN.DuellingDDQN,
    }
    replay_buffer = create_replay_buffer(args, obs_dim)
    timer         = Timer()

    evaluations = []
    episode_num = 0
    policy      = policies[args.policy](obs, num_actions, args)
    train(
        args, policy, evaluations, env, eval_env, replay_buffer, episode_num,
        timer, results_path, video_path, td_path, writer
    )



if __name__ == "__main__":
    main()
