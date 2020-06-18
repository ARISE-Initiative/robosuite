import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/bottle.xml"), name=name, joints=joints)


class CanObject(MujocoXMLObject):
    """
    Coke can object (used in PickPlace)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/can.xml"), name=name, joints=joints)


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/lemon.xml"), name=name, joints=joints)


class MilkObject(MujocoXMLObject):
    """
    Milk carton object (used in PickPlace)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/milk.xml"), name=name, joints=joints)


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in PickPlace)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/bread.xml"), name=name, joints=joints)


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/cereal.xml"), name=name, joints=joints)


class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/square-nut.xml"), name=name, joints=joints)


class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/round-nut.xml"), name=name, joints=joints)


class MilkVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in PickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/milk-visual.xml"), name=name, joints=joints)


class BreadVisualObject(MujocoXMLObject):
    """
    Visual fiducial of bread loaf (used in PickPlace)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/bread-visual.xml"), name=name, joints=joints)


class CerealVisualObject(MujocoXMLObject):
    """
    Visual fiducial of cereal box (used in PickPlace)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/cereal-visual.xml"), name=name, joints=joints)


class CanVisualObject(MujocoXMLObject):
    """
    Visual fiducial of coke can (used in PickPlace)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/can-visual.xml"), name=name, joints=joints)


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in PegInHole)
    """

    def __init__(self, name=None, joints=None):
        super().__init__(xml_path_completion("objects/plate-with-hole.xml"), name=name, joints=joints)


class DoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)
    """
    def __init__(self, name=None, joints=None, friction=None, damping=None, lock=False):
        xml_path = "objects/door.xml"
        if lock:
            xml_path = "objects/door_lock.xml"
        super().__init__(xml_path_completion(xml_path), name=name, joints=joints)
        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        collision = self.worldbody.find("./body/body[@name='collision']")
        node = collision.find("./body[@name='frame']")
        node = node.find("./body[@name='door']")
        hinge = node.find("./joint[@name='door_hinge']")
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        hinge = self._base_body.find("./joint[@name='door_hinge']")
        collision = self.worldbody.find("./body/body[@name='collision']")
        node = collision.find("./body[@name='frame']")
        node = node.find("./body[@name='door']")
        hinge = node.find("./joint[@name='door_hinge']")
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def _base_body(self):
        node = self.worldbody.find("./body[@name='door_body']")
        node = node.find("./body[@name='collision']")
        node = node.find("./body[@name='frame']")
        node = node.find("./body[@name='door']")
        return node
