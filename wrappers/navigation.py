import numpy as np
from gymnasium.core import Env
from gymnasium import spaces


class NavigationND(Env):
    def __init__(self, name='Navigation2D-v0'):
        super().__init__()
        self.name              = name
        self.dimension         = int(name.replace('Navigation', '').replace('D-v0', ''))
        self._time_limit       = 1000
        obs_high               = np.inf * np.ones(self.dimension)
        act_high               = np.ones(self.dimension)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high)
        self.action_space      = spaces.Box(low=-act_high, high=act_high)


    def reset(self, seed=None):
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")
        self.obs     = np.random.normal(loc=0.0, scale=1.0, size=self.dimension)
        # self._target = np.random.normal(loc=0.0, scale=1.0, size=self.dimension)
        # self.obs     = 10 * np.ones(self.dimension)
        # self.obs = np.zeros(self.dimension)

        self._target = np.zeros(self.dimension)
        self._step   = self._time_limit
        obs          = np.copy(self.obs)
        info         = {}
        return obs, info


    def step(self, action):
        dist_before = np.linalg.norm(self.obs - self._target)
        self.obs   += action
        dist_after  = np.linalg.norm(self.obs - self._target)
        obs         = np.copy(self.obs)
        reward      = (dist_before - dist_after) / np.sqrt(self.dimension)
        # reward      = - dist_after / (100 * np.sqrt(self.dimension))
        # reward      = - dist_after / np.sqrt(self.dimension)
        # reward      = - np.linalg.norm(action) / np.sqrt(self.dimension)
        term        = False
        if self._step == 1:
            trunc       = True
        else:
            trunc       = False
            self._step -= 1
        info        = {}
        return obs, reward, term, trunc, info
