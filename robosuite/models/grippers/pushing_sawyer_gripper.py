"""
A version of SawyerGripper but always closed.
"""
import numpy as np
from robosuite.models.grippers.sawyer_gripper import SawyerGripper


class PushingSawyerGripper(SawyerGripper):
    """
    Same as SawyerGripper, but always closed
    """

    def format_action(self, action):
        return np.array([1, -1])

    @property
    def dof(self):
        return 1
