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
    if renderer == 'nvisii':
        fname = 'config/nvisii_config.json'                
    elif renderer == 'igibson':
        fname = 'config/igibson_config.json'
    else:
        raise ValueError("renderer type can only be  'nvisii', or 'igibson'")

    dir_path = os.path.dirname(__file__)
    with open(os.path.join(dir_path, fname)) as f:
        config = json.load(f)

    return config

class Renderer():
    """
    Base class for all robosuite renderers
    Defines basic interface for all renderers to adhere to
    """

    def __init__(self, 
                 env,
                 renderer_type="default"):
        self.env = env        
        self.renderer_type = renderer_type

    def __str__(self):
        return f'<RendererObject renderer_type="{self.renderer_type}">'

    @abc.abstractmethod
    def render(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self): 
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_pixel_obs(self):
        raise NotImplementedError
