"""
This file contains the base renderer class for Mujoco environments.
"""

import abc

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
    def update(self, action): 
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
