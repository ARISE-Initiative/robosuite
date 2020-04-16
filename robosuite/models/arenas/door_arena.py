import numpy as np
from robosuite.models.arenas import Arena, TableArena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


class DoorArena(TableArena):
    """Workspace that contains an empty table."""

    def __init__(
        self, table_full_size=(0.8, 0.8, 0.8), table_friction=(1, 0.005, 0.0001)
    ):
        """
        Args:
            table_full_size: full dimensions of the table
            friction: friction parameters of the table
        """
        Arena.__init__(self, xml_path_completion("arenas/door_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")

        self.configure_location()
