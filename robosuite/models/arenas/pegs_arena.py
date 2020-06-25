import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


class PegsArena(Arena):
    """Workspace that contains a tabletop with two fixed pegs."""

    def __init__(
        self, table_full_size=(0.45, 0.69, 0.82), table_friction=(1, 0.005, 0.0001)
    ):
        """
        Args:
            table_full_size: full dimensions of the table
            table_friction: friction parameters of the table
        """
        super().__init__(xml_path_completion("arenas/pegs_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.table_body = self.worldbody.find("./body[@name='bin1']")
        self.peg1_body = self.worldbody.find("./body[@name='peg1']")
        self.peg2_body = self.worldbody.find("./body[@name='peg2']")
        # self.table_collision = self.table_body.find("./geom[@name='table_collision']")

        self.configure_location()

    def configure_location(self):
        self.bottom_pos = np.array([0, 0, 0])
        self.floor.set("pos", array_to_string(self.bottom_pos))

    @property
    def table_top_abs(self):
        """
        Returns the absolute position of table top.
        """
        return string_to_array(self.table_body.get("pos"))
