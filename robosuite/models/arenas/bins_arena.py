import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


class BinsArena(Arena):
    """Workspace that contains two bins placed side by side."""

    def __init__(
        self, bin1_pos=(0.1, -0.5, 0.8), table_full_size=(0.39, 0.49, 0.82), table_friction=(1, 0.005, 0.0001)
    ):
        """
        Args:
            table_full_size: full dimensions of the table
            friction: friction parameters of the table
        """
        super().__init__(xml_path_completion("arenas/bins_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.bin1_body = self.worldbody.find("./body[@name='bin1']")
        self.bin2_body = self.worldbody.find("./body[@name='bin2']")
        self.table_top_abs = np.array(bin1_pos)

        self.configure_location()

    def configure_location(self):
        self.bottom_pos = np.array([0, 0, 0])
        self.floor.set("pos", array_to_string(self.bottom_pos))
