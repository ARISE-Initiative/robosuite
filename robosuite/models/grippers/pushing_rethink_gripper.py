"""
A version of RethinkGripper but always closed.
"""
import numpy as np
from robosuite.models.grippers.rethink_gripper import RethinkGripper


class PushingRethinkGripper(RethinkGripper):
    """
    Same as RethinkGripper, but always closed
    """

    def format_action(self, action):
        return np.array([1, -1])

    @property
    def dof(self):
        return 1
