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
                Color (Relative / Absolute)
                Pattern

            Camera Randomization
                Position (Relative)
                Rotation (Relative)

            Lighting Randomization
                Position (Relative)
                Direction (Relative)
                Color (Relative)
                    Specular
                    Ambient
                    Diffuse
                Active
                CastShadow
                    TODO: do we really want this? excluding for now...

        Low-Dimension

            Pose Noise
            Joint Noise
            Velocity Noise

            TODO (requires observation categorization, too much of a pain for now...)
                -Perhaps only noise up the proprioception??? (and assume fixed set of keys)
        
    Action Noise

        TODO

    Dynamics Parameter Noise

        TODO

Other options:
    Frequency (how many times per step? beginning of each episode?)
    Option to perform all randomization operations (disregarding the frequency)


TODO: ability to restore to unmodified model / defaults
TODO: why can we still see things with all the lights off?
TODO: ability to apply randomization from an mjstate / mjmodel pair?
    (so that these randomizations can also occur in a replay buffer)
TODO: each composite object should specify texture groups that get randomized together
    (geom groups per texture)
TODO: do all materials need to be whitened? or only those in geom list
TODO: color perturbations could probably be improved by using a space other than RGB
"""
import numpy as np

from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder, LightingModder, CameraModder


class DomainRandomizationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.tex_modder = TextureModder(
            sim=self.env.sim,
            geom_names=None, # all geoms are randomized
            randomize_local=True,
            local_rgb_interpolation=0.2,
            texture_variations=['rgb', 'checker', 'noise', 'gradient'],
            randomize_skybox=False,
        )

        self.light_modder = LightingModder(
            sim=self.env.sim,
            light_names=None, # all lights are randomized
            randomize_position=True,
            randomize_direction=True,
            randomize_specular=True,
            randomize_ambient=True,
            randomize_diffuse=True,
            randomize_active=True,
            position_perturbation_size=0.1,
            direction_perturbation_size=0.35,
            specular_perturbation_size=0.1,
            ambient_perturbation_size=0.1,
            diffuse_perturbation_size=0.1,
        )

        self.camera_modder =  CameraModder(
            sim=self.env.sim,
            camera_names=None, # all cameras are randomized
            randomize_position=True,
            randomize_rotation=True,
            randomize_fovy=True,
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
        self.tex_modder.randomize()
        # self.camera_modder.randomize()
        # self.light_modder.randomize()

    def save_default_domain(self):
        pass

    def restore_default_domain(self):
        pass

