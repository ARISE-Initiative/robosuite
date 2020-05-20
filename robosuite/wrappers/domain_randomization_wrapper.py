"""
This file implements a wrapper for facilitating domain randomization over
robosuite environments.

TODO: make list of arguments for DRWrapper to support

Categories of DR:

    Observation Noise

        All

            Sensor Dropout

        RGB Image

            Texture Randomization
                Color
                    Local / Global
                Pattern

            Camera Randomization
                Position
                    Local / Global
                Rotation
                    Local / Global

            Lighting Randomization
                Position
                    Local / Global
                Direction
                    Local / Global
                Color

        Low-Dimension

            Pose Noise
            Joint Noise
            Velocity Noise
        
    Action Noise

        TODO

    Dynamics Parameter Noise

        TODO

Other options:
    Frequency (how many times per step? beginning of each episode?)
    Option to perform all randomization operations (disregarding the frequency)


TODO: ability to restore to unmodified model / defaults
TODO: ability to apply randomization from an mjstate / mjmodel pair?
    (so that these randomizations can also occur in a replay buffer)
TODO: each composite object should specify texture groups that get randomized together
    (geom groups per texture)
"""
import numpy as np

from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder, LightingModder, CameraModder


class DomainRandomizationWrapper(Wrapper):
    def __init__(self, env, path=None):
        super().__init__(env)
        self.tex_modder = TextureModder(self.env.sim, path=path)
        self.light_modder = LightingModder(self.env.sim)
        self.camera_modder =  CameraModder(
            sim=self.env.sim, 
            camera_names=self.env.camera_names,
            perturb_position=True,
            perturb_rotation=True,
            perturb_fovy=True,
            position_perturbation_size=0.01,
            rotation_perturbation_size=0.087,
            fovy_perturbation_size=5.,
        )
        self.save_default_domain()

    def reset(self):
        self.restore_default_domain()
        ret = super().reset()
        self.save_default_domain()

        ### TODO: randomization here? with condition check ###
        self.randomize_domain()
        return ret

    def step(self, action):
        ### TODO: randomization here? with condition check ###
        self.randomize_domain()

        return super().step(action)

    def randomize_domain(self):
        self.randomize_texture()
        self.randomize_camera()

    def save_default_domain(self):
        pass

    def restore_default_domain(self):
        pass

    def randomize_all(self):
        self.randomize_camera()
        self.randomize_texture()
        self.randomize_light()
        # self.randomize_material()

    def randomize_texture(self):
        self.tex_modder.randomize()

    def randomize_light(self):
        self.light_modder.randomize()

    # def randomize_material(self):
    #     self.mat_modder.randomize()

    def randomize_camera(self):
        self.camera_modder.randomize()