"""
This file implements a wrapper for facilitating domain randomization over
robosuite environments.
"""

from robosuite.wrappers import Wrapper

class DRWrapper(Wrapper):
    env = None

    def __init__(self, env):
        super().__init__(env=env)
