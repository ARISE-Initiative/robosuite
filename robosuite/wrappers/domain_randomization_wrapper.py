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
        self.randomize_texture()
        self.randomize_light()
        self.randomize_material()
        self.randomize_camera()

    def randomize_texture(self):
        self.tex_modder.randomize()

    def randomize_light(self):
        self.light_modder.randomize()

    def randomize_material(self):
        self.mat_modder.randomize()

    def randomize_camera(self):
        self.camera_modder.randomize()
