from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import xml_path_completion


class MujocoWorldBase(MujocoXML):
    """Base class to inherit all mujoco worlds from."""

    def __init__(self):
        super().__init__(xml_path_completion("base.xml"))
