import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/bottle.xml"),
                         name=name, joints=joints, obj_type="all", duplicate_collision_geoms=True)


class CanObject(MujocoXMLObject):
    """
    Coke can object (used in PickPlace)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/can.xml"),
                         name=name, joints=joints, obj_type="all", duplicate_collision_geoms=True)


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/lemon.xml"),
                         name=name, joints=joints, obj_type="all", duplicate_collision_geoms=True)


class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/milk.xml"),
                         name=name, joints=joints, obj_type="all", duplicate_collision_geoms=True)


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in PickPlace)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/bread.xml"),
                         name=name, joints=joints, obj_type="all", duplicate_collision_geoms=True)


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/cereal.xml"),
                         name=name, joints=joints, obj_type="all", duplicate_collision_geoms=True)


class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/square-nut.xml"),
                         name=name, joints=joints, obj_type="all", duplicate_collision_geoms=True)


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/round-nut.xml"),
                         name=name, joints=joints, obj_type="all", duplicate_collision_geoms=True)


class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in PickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/milk-visual.xml"),
                         name=name, joints=joints, obj_type="visual", duplicate_collision_geoms=True)


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/bread-visual.xml"),
                         name=name, joints=joints, obj_type="visual", duplicate_collision_geoms=True)


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/cereal-visual.xml"),
                         name=name, joints=joints, obj_type="visual", duplicate_collision_geoms=True)


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in PickPlace)

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/can-visual.xml"),
                         name=name, joints=joints, obj_type="visual", duplicate_collision_geoms=True)


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in PegInHole)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/plate-with-hole.xml"),
                         name=name, joints=joints, obj_type="all", duplicate_collision_geoms=True)


class DoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """
    def __init__(self, name=None, joints=None, friction=None, damping=None, lock=False):
        xml_path = "objects/door.xml"
        if lock:
            xml_path = "objects/door_lock.xml"
        super().__init__(xml_path_completion(xml_path),
            name=name, joints=joints, obj_type="all", duplicate_collision_geoms=True)
        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        obj = self.worldbody.find("./body/body[@name='object']")
        node = obj.find("./body[@name='frame']")
        node = node.find("./body[@name='door']")
        hinge = node.find("./joint[@name='door_hinge']")
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        obj = self.worldbody.find("./body/body[@name='object']")
        node = obj.find("./body[@name='frame']")
        node = node.find("./body[@name='door']")
        hinge = node.find("./joint[@name='door_hinge']")
        hinge.set("damping", array_to_string(np.array([damping])))
