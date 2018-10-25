import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class Baxter(Robot):
    """Baxter is a hunky bimanual robot designed by Rethink Robotics."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/baxter/robot.xml"))

        self.bottom_offset = np.array([0, 0, -0.913])
        self.left_hand = self.worldbody.find(".//body[@name='left_hand']")

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 14

    @property
    def joints(self):
        out = []
        for s in ["right_", "left_"]:
            out.extend(s + a for a in ["s0", "s1", "e0", "e1", "w0", "w1", "w2"])
        return out

    @property
    def init_qpos(self):
        # Arms ready to work on the table
        return np.array([
            0.535, -0.093, 0.038, 0.166, 0.643, 1.960, -1.297,
            -0.518, -0.026, -0.076, 0.175, -0.748, 1.641, -0.158])

        # Arms half extended
        return np.array([
            0.752, -0.038, -0.021, 0.161, 0.348, 2.095, -0.531,
            -0.585, -0.117, -0.037, 0.164, -0.536, 1.543, 0.204])

        # Arms fully extended
        return np.zeros(14)
