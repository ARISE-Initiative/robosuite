"""
This file implements a wrapper for facilitating domain randomization over
robosuite environments.
"""

from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder, LightingModder, MaterialModder


class DRWrapper(Wrapper):
    env = None

    def __init__(self, env):
        super().__init__(env)
        self.reset()

    def reset(self):
        super().reset()
        # Env will be updated after reset
        self.tex_modder = TextureModder(self.env.sim)
        self.light_modder = LightingModder(self.env.sim)
        self.mat_modder = MaterialModder(self.env.sim)
        self.randomize_all()

    def randomize_all(self):
        for modder in (self.tex_modder, self.light_modder, self.mat_modder):
            modder.randomize()
