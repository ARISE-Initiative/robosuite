from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion


class DoorObject(MujocoXMLObject):
  """
  Door with handle
  """

  def __init__(self):
        super().__init__(xml_path_completion("objects/door_small.xml"))


class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bottle.xml"))


class CanObject(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/can.xml"))


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/lemon.xml"))


class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/milk.xml"))


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bread.xml"))


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/cereal.xml"))


class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in SawyerNutAssembly)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/square-nut.xml"))


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in SawyerNutAssembly)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/round-nut.xml"))


class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in SawyerPickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/milk-visual.xml"))


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/bread-visual.xml"))


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/cereal-visual.xml"))


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/can-visual.xml"))


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in BaxterPegInHole)
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/plate-with-hole.xml"))
