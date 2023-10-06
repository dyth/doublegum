import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class ConcatObs(gym.ActionWrapper):
    def __init__(self, env, num=1):
        super().__init__(env)
        self.num               = num
        self.length            = self.observation_space.low.shape[0]
        low                    = np.repeat(self.observation_space.low, self.num, axis=0)
        high                   = np.repeat(self.observation_space.high, self.num, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._get_obs(obs), reward, term, trunc, info

    def _get_obs(self, obs):
        obs = np.tile(obs, (self.num,))
        return obs

