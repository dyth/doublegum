import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class ConcatAction(gym.ActionWrapper):
    def __init__(self, env, num=1):
        super().__init__(env)
        self.num          = num
        self.length       = self.action_space.low.shape[0]
        low               = np.repeat(self.action_space.low, self.num, axis=0)
        high              = np.repeat(self.action_space.high, self.num, axis=0)
        self.action_space = Box(low=low, high=high, dtype=self.action_space.dtype)

    def action(self, action):
        action = action[:self.length,]
        return action

    def reverse_action(self, action):
        action = np.tile(action, (self.num,))
        return action
