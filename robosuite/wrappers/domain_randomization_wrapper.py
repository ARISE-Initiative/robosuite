"""
This file implements a wrapper for facilitating domain randomization over
robosuite environments.
"""
import numpy as np

from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder, LightingModder, CameraModder

DEFAULT_COLOR_ARGS = {
    'geom_names' : None, # all geoms are randomized
    'randomize_local' : True, # sample nearby colors
    'randomize_material' : True, # randomize material reflectance / shininess / specular
    'local_rgb_interpolation' : 0.2,
    'local_material_interpolation' : 0.3,
    'texture_variations' : ['rgb', 'checker', 'noise', 'gradient'], # all texture variation types
    'randomize_skybox' : True, # by default, randomize skybox too
}

DEFAULT_CAMERA_ARGS = {
    'camera_names' : None, # all cameras are randomized
    'randomize_position' : True,
    'randomize_rotation' : True,
    'randomize_fovy' : True,
    'position_perturbation_size' : 0.01,
    'rotation_perturbation_size' : 0.087,
    'fovy_perturbation_size' : 5.,
}

DEFAULT_LIGHTING_ARGS = {
    'light_names' : None, # all lights are randomized
    'randomize_position' : True,
    'randomize_direction' : True,
    'randomize_specular' : True,
    'randomize_ambient' : True,
    'randomize_diffuse' : True,
    'randomize_active' : True,
    'position_perturbation_size' : 0.1,
    'direction_perturbation_size' : 0.35,
    'specular_perturbation_size' : 0.1,
    'ambient_perturbation_size' : 0.1,
    'diffuse_perturbation_size' : 0.1,
}


class DomainRandomizationWrapper(Wrapper):
    """
    Wrapper that allows for domain randomization mid-simulation.

    Args:
        env (MujocoEnv): The environment to wrap.

        seed (int): Integer used to seed all randomizations from this wrapper. It is
            used to create a np.random.RandomState instance to make sure samples here
            are isolated from sampling occurring elsewhere in the code. If not provided,
            will default to using global random state.

        randomize_color (bool): if True, randomize geom colors and texture colors

        randomize_camera (bool): if True, randomize camera locations and parameters

        randomize_lighting (bool): if True, randomize light locations and properties

        color_randomization_args (dict): Color-specific randomization arguments

        camera_randomization_args (dict): Camera-specific randomization arguments

        lighting_randomization_args (dict): Lighting-specific randomization arguments

        randomize_on_reset (bool): if True, randomize on every call to @reset. This, in
            conjunction with setting @randomize_every_n_steps to 0, is useful to
            generate a new domain per episode.

        randomize_every_n_steps (int): determines how often randomization should occur. Set
            to 0 if randomization should happen manually (by calling @randomize_domain)

    """
    def __init__(
        self, 
        env,
        seed=None,
        randomize_color=True,
        randomize_camera=True,
        randomize_lighting=True,
        color_randomization_args=DEFAULT_COLOR_ARGS,
        camera_randomization_args=DEFAULT_CAMERA_ARGS,
        lighting_randomization_args=DEFAULT_LIGHTING_ARGS,
        randomize_on_reset=True,
        randomize_every_n_steps=1,
    ):
        super().__init__(env)

        self.seed = seed
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = None
        self.randomize_color = randomize_color
        self.randomize_camera = randomize_camera
        self.randomize_lighting = randomize_lighting
        self.color_randomization_args = color_randomization_args
        self.camera_randomization_args = camera_randomization_args
        self.lighting_randomization_args = lighting_randomization_args
        self.randomize_on_reset = randomize_on_reset
        self.randomize_every_n_steps = randomize_every_n_steps

        self.step_counter = 0

        self.modders = []

        if self.randomize_color:
            self.tex_modder = TextureModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.color_randomization_args
            )
            self.modders.append(self.tex_modder)

        if self.randomize_camera:
            self.camera_modder =  CameraModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.camera_randomization_args,
            )
            self.modders.append(self.camera_modder)

        if self.randomize_lighting:
            self.light_modder = LightingModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.lighting_randomization_args,
            )
            self.modders.append(self.light_modder)

        self.save_default_domain()

    def reset(self):
        """
        Extends superclass method to reset the domain randomizer.

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # undo all randomizations
        self.restore_default_domain()

        # normal env reset
        ret = super().reset()

        # save the original env parameters
        self.save_default_domain()

        # reset counter for doing domain randomization at a particular frequency
        self.step_counter = 0

        if self.randomize_on_reset:
            # domain randomize + regenerate observation
            self.randomize_domain()
            ret = self.env._get_observation()

        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate domain randomization

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """

        # functionality for randomizing at a particular frequency
        if self.randomize_every_n_steps > 0:
            if self.step_counter % self.randomize_every_n_steps == 0:
                self.randomize_domain()
        self.step_counter += 1

        return super().step(action)

    def randomize_domain(self):
        """
        Runs domain randomization over the environment.
        """
        for modder in self.modders:
            modder.randomize()

    def save_default_domain(self):
        """
        Saves the current simulation model parameters so
        that they can be restored later.
        """
        for modder in self.modders:
            modder.save_defaults()

    def restore_default_domain(self):
        """
        Restores the simulation model parameters saved
        in the last call to @save_default_domain.
        """
        for modder in self.modders:
            modder.restore_defaults()
