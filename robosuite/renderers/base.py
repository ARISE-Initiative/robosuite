"""
This file contains the base renderer class for Mujoco environments.
"""

import abc
import json
import os


def load_renderer_config(renderer):
    """Loads the config of the specified renderer.
    Modify the dictionary returned by this function
    according to reuirements.

    Args:
        renderer (str): Name of the renderer to use.

    Returns:
        dict: renderer default config.
    """
    if renderer == "nvisii":
        fname = "config/nvisii_config.json"
    elif renderer == "mjviewer":
        return {}
    else:
        raise ValueError(f"renderer type can only be  'nvisii', or 'mjviewer' got '{renderer}'")

    dir_path = os.path.dirname(__file__)
    with open(os.path.join(dir_path, fname)) as f:
        config = json.load(f)

    return config


class Renderer:
    """
    Base class for all robosuite renderers
    Defines basic interface for all renderers to adhere to
    """

    def __init__(self, env, renderer_type="mujoco"):
        self.env = env
        self.renderer_type = renderer_type

    def __str__(self):
        """Prints the renderer type in a formatted way

        Returns:
            str: string representing the renderer
        """
        return f'<RendererObject renderer_type="{self.renderer_type}">'

    @abc.abstractmethod
    def render(self, **kwargs):
        """Renders the current state with the specified renderer"""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self):
        """Updates the states in the renderer (for NVISII)"""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """Closes the renderer objects"""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Reset the renderer with initial states for environment"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_pixel_obs(self):
        """Get the pixel observations from the given renderer

        Returns:
            numpyarr: numpy array representing pixels of renderer
        """
        raise NotImplementedError
