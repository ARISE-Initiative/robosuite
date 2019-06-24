"""
This file implements a wrapper for facilitating domain randomization over
robosuite environments.
"""

from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder, LightingModder, MaterialModder, CameraModder


class DRWrapper(Wrapper):
    env = None

    def __init__(self, env):
        super().__init__(env)
        self.tex_modder = TextureModder(self.env.sim)
        self.light_modder = LightingModder(self.env.sim)
        self.mat_modder = MaterialModder(self.env.sim)
        self.camera_modder =  CameraModder(sim=self.env.sim, camera_name=self.env.camera_name)

    def reset(self):
        super().reset()
        # Env will be updated after reset
        self.tex_modder = TextureModder(self.env.sim)
        self.light_modder = LightingModder(self.env.sim)
        self.mat_modder = MaterialModder(self.env.sim)
        self.camera_modder =  CameraModder(sim=self.env.sim, camera_name=self.env.camera_name)
        self.randomize_all()

    def render(self, **kwargs):
        self.randomize_all()
        super().render(**kwargs)

    def randomize_all(self):
        for modder in (self.tex_modder, self.light_modder, self.mat_modder, self.camera_modder):
            modder.randomize()
