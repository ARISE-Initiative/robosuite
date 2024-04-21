import numpy as np

from robosuite.models.bases.leg_base_model import LegBaseModel
from robosuite.utils.mjcf_utils import xml_path_completion


class B1(LegBaseModel):
    """
    Rethink's Generic Mount (Officially used on Baxter).

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/b1/robot.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        return 0.1

    @property
    def init_qpos(self):
        return np.array([0.0, 0.9, -1.8] * 4)


class B1Floating(LegBaseModel):
    """
    Rethink's Generic Mount (Officially used on Baxter).

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/b1/robot.xml"), idn=idn)

        self._remove_joint_actuation("leg")
        self._remove_free_joint()

        self._add_mobile_joint()

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        return 0.1

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0])
