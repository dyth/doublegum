import os

import gymnasium as gym
import imageio
import numpy as np

from wrappers.common import TimeStep


# Taken from
# https://github.com/denisyarats/pytorch_sac/
class VideoRecorder(gym.Wrapper):

    def __init__(self,
                 env: gym.Env,
                 save_folder: str = '',
                 height: int = 128,
                 width: int = 128,
                 fps: int = 30,
                 camera_id: int = -1):
        super().__init__(env)

        self.current_episode = 0
        self.save_folder = save_folder
        self.height = height
        self.width = width
        self.fps = fps
        # self.camera_id = camera_id
        self.camera_id = 2 if env.domain_name == 'quadruped' else 0

        self.frames = []
        self.reward = 0.

        try:
            os.makedirs(save_folder, exist_ok=True)
        except:
            pass

    def step(self, action: np.ndarray) -> TimeStep:

        frame = self.env.render(mode='rgb_array',
                                height=self.height,
                                width=self.width,
                                camera_id=self.camera_id)

        if frame is None:
            try:
                frame = self.sim.render(width=self.width,
                                        height=self.height,
                                        mode='offscreen')
                frame = np.flipud(frame)
            except:
                raise NotImplementedError('Rendering is not implemented.')

        self.frames.append(frame)

        observation, reward, term, trunc, info = self.env.step(action)
        self.reward += reward

        if term or trunc:
            save_file = os.path.join(self.save_folder,
                                     f'{self.current_episode}_return{self.reward:.4f}.mp4')
            imageio.mimsave(save_file, self.frames, fps=self.fps)
            self.frames = []
            self.current_episode += 1
            self.reward = 0.

        return observation, reward, term, trunc, info
