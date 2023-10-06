"""
This is adapted from https://github.com/ikostrikov/jaxrl

Robosuite adapted from
https://robosuite.ai/docs/algorithms/benchmarking.html
https://github.com/ARISE-Initiative/robosuite-benchmark/util/rlkit_utils.py#L31 for suite.make command
https://github.com/ARISE-Initiative/robosuite-benchmark/util/arguments.py#L23 for the suite make args
"""
import gymnasium as gym
import metaworld
import random
import robosuite as suite
from robosuite.controllers import load_controller_config
from gymnasium.wrappers import RescaleAction
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from typing import Optional

import wrappers


def make_robosuite(env_name):
    env_name = env_name.split('_')[1]

    # robots = components[1].split('-')

    # components = env_name.split('_')

    # env_name   = '_'.join(components[4:])

    # robots     = components[1].split('-')

    # controller                           = '_'.join(components[2:4]).split('-')
    # controller_configs                   = load_controller_config(default_controller=controller[0])
    # controller_configs['impedance_mode'] = controller[1]

    env = wrappers.OldToNewGym(
        wrappers.RoboSuiteWrapper(
            suite.make(
                env_name=env_name,
                # robots=robots,
                robots='Panda',
                horizon=500,
                control_freq=20,  # Hz
                reward_scale=1.0,
                hard_reset=True,
                ignore_done=True,

                has_renderer=False,
                has_offscreen_renderer=False,
                use_object_obs=True,
                use_camera_obs=False,
                reward_shaping=True,
                controller_configs=load_controller_config(default_controller='OSC_POSE'),
                # controller_configs=load_controller_config(default_controller='JOINT_VELOCITY'),
                # controller_configs=controller_configs,
            )
        ),
        duration=500
    )
    return env


def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True,
             terminate_when_unhealthy: bool = True,
             action_concat: int = 1,
             obs_concat: int = 1,
             continuous: bool = True,
             ) -> gym.Env:

    # Check if the env is in gym.
    env_ids = list(gym.envs.registry.keys())

    if env_name in env_ids:
        env = gym.make(env_name)
        # env = gym.make(env_name, terminate_when_unhealthy=False)
        save_folder = None

    elif 'Navigation' in env_name:
        env = wrappers.NavigationND(env_name)

    elif 'metaworld' in env_name:
        env_name = '_'.join(env_name.split('_')[1:])
        mt1 = metaworld.MT1(env_name, seed=0)
        env = mt1.train_classes[env_name]()
        task = random.choice(mt1.train_tasks)
        env.set_task(task)
        env = wrappers.OldToNewGym(env, duration=env.max_path_length)
        save_folder = None

    elif 'robosuite' in env_name:
        env = make_robosuite(env_name)
        save_folder = None

    else:
        domain_name, task_name = env_name.split('-')
        env = wrappers.DMCEnv(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
        env = wrappers.FlattenAction(env)

    # if add_episode_monitor:
    #     env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    if continuous:
        env = RescaleAction(env, -1.0, 1.0)

    # if action_concat > 1:
    #     env = wrappers.ConcatAction(env, action_concat)
    # if obs_concat > 1:
    #     env = wrappers.ConcatObs(env, obs_concat)

    # if save_folder is not None:
    #     env = wrappers.VideoRecorder(env, save_folder=save_folder)

    # TODO: should probably move this inside DMC
    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            camera_id = 2 if domain_name == 'quadruped' else 0
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only,
                                      render_kwargs={
                                          'pixels': {
                                              'height': image_size,
                                              'width': image_size,
                                              'camera_id': camera_id
                                          }
                                      })
        env = wrappers.TakeKey(env, take_key='pixels')
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
