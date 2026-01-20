import xml.etree.ElementTree as ET
import robosuite.macros as macros
from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import convert_to_string, find_elements, xml_path_completion


class MujocoWorldBase(MujocoXML):
    """Base class to inherit all mujoco worlds from."""

    def __init__(self, enable_multiccd=False, enable_sleeping_islands=False):
        super().__init__(xml_path_completion("base.xml"))
        # Modify the simulation timestep to be the requested value
        options = find_elements(root=self.root, tags="option", attribs=None, return_first=True)
        options.set("timestep", convert_to_string(macros.SIMULATION_TIMESTEP))
        self.enable_multiccd = enable_multiccd
        self.enable_sleeping_islands = enable_sleeping_islands
        if self.enable_multiccd:
            multiccd_elem = ET.fromstring(
                """<option> <flag multiccd="enable"/> </option>"""
            )
            mujoco_elem = find_elements(self.root, "mujoco")
            mujoco_elem.insert(0, multiccd_elem)
        if self.enable_sleeping_islands:
            sleeping_elem = ET.fromstring(
                """<option> <flag sleep="enable"/> </option>"""
            )
            mujoco_elem = find_elements(self.root, "mujoco")
            mujoco_elem.insert(0, sleeping_elem)   
        
