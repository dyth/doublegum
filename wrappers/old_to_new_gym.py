'''
Adapted from https://github.com/tseyde/decqn/commit/db660bf9c3d89784b2bd7552cace5761ec68f086
'''
import gymnasium as gym
from gymnasium import spaces


class OldToNewGym(gym.Wrapper):
    def __init__(self, env, duration=1000):
        super().__init__(env)
        self._steps = duration
        self._step = None
        self.observation_space = spaces.Box(low=env.observation_space.low, high=env.observation_space.high)
        self.action_space = spaces.Box(low=env.action_space.low, high=env.action_space.high)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        self._step = 0
        return obs, {}

    def step(self, action):
        """returns a tuple of obs, reward, term, trunc, info"""
        obs, reward, term, info = self.env.step(action)
        if self._step + 1 > self._steps - 1:
            """Manually reset after fixed no of timesteps: https://github.com/rlworkgroup/metaworld/issues/236"""
            trunc = True
        else:
            trunc = False
        self._step += 1
        return obs, reward, term, trunc, info
