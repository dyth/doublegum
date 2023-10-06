import argparse
import numpy as np


def type_bool(x):
    return x.lower() != 'false'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",             default=0,          type=int)
    parser.add_argument("--get-random",       default=False,      type=type_bool) # rollout a random policy
    parser.add_argument("--log-td",           default=False,      type=type_bool)
    parser.add_argument("--evaluate",         default=True,       type=type_bool)
    parser.add_argument("--eval-episodes",    default=10,         type=int)
    parser.add_argument("--discount",         default=0.99,       type=float)     # Discount factor
    parser.add_argument("--jit",              default=True,       type=type_bool) # Use jit compilation

    parser.add_argument("--name",             default="")
    parser.add_argument("--folder",           default="./results")

    parser.add_argument("--video",            default=False,      type=type_bool) # save video
    parser.add_argument("--log-freq",         default=1e4,        type=int)
    parser.add_argument("--video-freq",       default=5e5,        type=int)
    parser.add_argument("--get-stats",        default=False,      type=type_bool)

    # noise
    parser.add_argument("--explore",          default=True,       type=type_bool) # add noise to exploration or not
    parser.add_argument("--expl-noise",       default=.2,         type=float)     # Std of Gaussian exploration noise, adapted from URLB's Twin-DDPG
    parser.add_argument("--critic-noise",     default=1.,         type=float)     # bootstrapping sampling from actor
    parser.add_argument("--actor-noise",      default=1.,         type=float)     # sampling from actor for actor loss

    # replay buffer configuration
    parser.add_argument("--nstep",            default=1,          type=int)
    parser.add_argument("--replay-ratio",     default=1,          type=int)
    parser.add_argument("--replay-size",      default=1e6,        type=int)
    parser.add_argument("--random-sampling",  default=True,       type=type_bool)
    parser.add_argument("--batch-size",       default=256,        type=int)

    # EMA
    parser.add_argument("--tau",              default=0.005,      type=float)     # Critic target network update rate
    return parser


def parser_cont():
    parser = parse_args()
    parser.add_argument("--policy",           default="DDPG")
    parser.add_argument("--env",              default="cheetah-run")         # gym environment name

    # evaluation
    parser.add_argument("--start-timesteps",  default=1e4,        type=int)       # Time steps initial random policy is used TD3 (25e3)
    parser.add_argument("--max-timesteps",    default=1e6,        type=int)       # Max time steps to run environment
    parser.add_argument("--eval-freq",        default=1e4,        type=int)

    # architectural details
    parser.add_argument("--actor-lr",         default=3e-4,       type=float)
    parser.add_argument("--critic-lr",        default=3e-4,       type=float)
    parser.add_argument("--actor",            default=[256, 256], type=int,       nargs="+")
    parser.add_argument("--critic",           default=[256, 256], type=int,       nargs="+")
    parser.add_argument("--actor-groups",     default=16,         type=int)
    parser.add_argument("--critic-groups",    default=16,         type=int)
    parser.add_argument("--critic-sn",        default=False,      type=type_bool)
    parser.add_argument("--actor-sn",         default=False,      type=type_bool)
    parser.add_argument("--actor-tau",        default=1.,         type=float)     # Actor target network update rate

    # pessimism factor for DoubleGum
    parser.add_argument("--c",                default=0.0,        type=float)

    # Temperature of XQL
    parser.add_argument("--beta",             default=3.0,        type=float)

    # MoG-DDPG params
    parser.add_argument("--num-components",   default=5,          type=int)       # number of components in mixture model
    parser.add_argument("--init-scale",       default=1e-3,       type=float)     # scale of standard deviation

    # Twin Networks or without (Twin Networks are run by default here)
    parser.add_argument("--ensemble",         default=2,          type=int)
    parser.add_argument("--actor-ensemble",   default=1,          type=int)
    parser.add_argument("--pessimism",        default=0,          type=int)
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = np.random.randint(10000)
    np.random.seed(args.seed)

    if args.pessimism > args.ensemble:
        raise ValueError(f"args.pessimism ({args.pessimism}) should not be larger than args.ensemble ({args.ensemble})")

    # args.eval_freq = int(args.max_timesteps / 100)
    # args.video_freq = int(args.max_timesteps / 2)
    return args


def parser_disc():
    parser = parse_args()
    parser.add_argument("--policy",           default="DQN")
    parser.add_argument("--env",              default="CartPole-v1")         # gym environment name

    # evaluation
    parser.add_argument("--start-timesteps",  default=2e3,        type=int)       # Time steps initial random policy is used TD3 (25e3)
    parser.add_argument("--max-timesteps",    default=1e5,        type=int)       # Max time steps to run environment
    parser.add_argument("--eval-freq",        default=1e3,        type=int)

    # architectural details
    parser.add_argument("--network-lr",       default=3e-4,       type=float)
    parser.add_argument("--network",          default=[256, 256], type=int, nargs="+")
    parser.add_argument("--network-groups",   default=0,          type=int)
    parser.add_argument("--network-sn",       default=False,      type=type_bool)
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = np.random.randint(10000)
    np.random.seed(args.seed)

    return args
