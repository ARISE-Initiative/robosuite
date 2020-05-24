"""
Defines a string based method of initializing grippers
"""
from .panda_gripper import PandaGripper
from .pr2_gripper import PR2Gripper
from .rethink_gripper import RethinkGripper
from .robotiq_85_gripper import Robotiq85Gripper
from .robotiq_three_finger_gripper import RobotiqThreeFingerGripper
from .jaco_three_finger_gripper import JacoThreeFingerGripper
from .robotiq_140_gripper import Robotiq140Gripper
from .null_gripper import NullGripper


def gripper_factory(name, idn=0):
    """
    Generator for grippers

    Creates a GripperModel instance with the provided name.

    Args:
        name: the name of the gripper class
        idn: idn (int or str): Number or some other unique identification string for this gripper instance

    Returns:
        gripper: GripperModel instance

    Raises:
        XMLError: [description]
    """
    if name == "RethinkGripper":
        return RethinkGripper(idn=idn)
    if name == "PR2Gripper":
        return PR2Gripper(idn=idn)
    if name == "Robotiq85Gripper":
        return Robotiq85Gripper(idn=idn)
    if name == "RobotiqThreeFingerGripper":
        return RobotiqThreeFingerGripper(idn=idn)
    if name == "PandaGripper":
        return PandaGripper(idn=idn)
    if name == "JacoThreeFingerGripper":
        return JacoThreeFingerGripper(idn=idn)
    if name == "Robotiq140Gripper":
        return Robotiq140Gripper(idn=idn)
    if name is None:
        return NullGripper(idn=idn)
    raise ValueError("Unknown gripper name: {}".format(name))
