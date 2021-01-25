from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import xml_path_completion, convert_to_string, find_elements
import robosuite.utils.macros as macros


class MujocoWorldBase(MujocoXML):
    """Base class to inherit all mujoco worlds from."""

    def __init__(self):
        super().__init__(xml_path_completion("base.xml"))
        # Modify the simulation timestep to be the requested value
        options = find_elements(root=self.root, tags="option", attribs=None, return_first=True)
        options.set("timestep", convert_to_string(macros.SIMULATION_TIMESTEP))
