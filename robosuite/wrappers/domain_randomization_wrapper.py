"""
This file implements a wrapper for facilitating domain randomization over
robosuite environments.

TODO: decide if we want robot to support per-geom randomization or randomize as a whole group
TODO: put textures into grippers as well
TODO: put textures on generated objects
TODO: option to turn texture randomization on vs. off (or default to geom color rand)
TODO: make list of arguments for DRWrapper to support
TODO: ability to restore to unmodified model / defaults
TODO: modify wrapper settings on demand (to make it easy for an external curriculum)
TODO: support action noise
TODO: support camera rotations + multiple cameras
TODO: support image translation
TODO: support sensor delay / repeat / dropout
"""
import numpy as np

from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder, LightingModder, MaterialModder, CameraModder


class DomainRandomizationWrapper(Wrapper):
    def __init__(self, env, random_state=None, path=None):
        super().__init__(env)
        # self.action_noise = 1  # TODO: Should this be argument
        self.random_state = random_state
        self.tex_modder = TextureModder(self.env.sim, random_state=random_state, path=path)
        self.light_modder = LightingModder(self.env.sim, random_state=random_state)
        self.mat_modder = MaterialModder(self.env.sim, random_state=random_state)
        self.camera_modder =  CameraModder(sim=self.env.sim, camera_name=self.env.camera_names[0], random_state=random_state)

    def reset(self, random_state=None):
        if random_state is not None:
            self.set_random_state(random_state)
        return super().reset()

    def set_random_state(self, random_state=None):
        self.random_state = int(random_state)
        self.tex_modder = TextureModder(self.env.sim, random_state=self.random_state)
        self.light_modder = LightingModder(self.env.sim, random_state=self.random_state)
        self.mat_modder = MaterialModder(self.env.sim, random_state=self.random_state)
        self.camera_modder = CameraModder(sim=self.env.sim, camera_name=self.env.camera_names[0], random_state=self.random_state)

    def step(self, action):
        #action += np.random.normal(scale=self.action_noise, size=action.shape)
        # self.randomize_all()
        self.randomize_texture()
        return super().step(action)

    def randomize_all(self):
        self.randomize_camera()
        self.randomize_texture()
        self.randomize_light()
        self.randomize_material()

    def randomize_texture(self):
        self.tex_modder.randomize()

    def randomize_light(self):
        self.light_modder.randomize()

    def randomize_material(self):
        self.mat_modder.randomize()

    def randomize_camera(self):
        self.camera_modder.randomize()