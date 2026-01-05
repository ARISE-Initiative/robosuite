from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion


class YamStationArena(Arena):
    """
    Workspace that contains the Yam Station (gate, table, cameras).
    """

    def __init__(self):
        super().__init__(xml_path_completion("arenas/yam_station.xml"))




