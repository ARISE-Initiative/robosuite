"""
Defines a string based method of initializing grippers
"""
from .two_finger_gripper import TwoFingerGripper, LeftTwoFingerGripper
from .pr2_gripper import PR2Gripper
from .robotiq_gripper import RobotiqGripper
from .pushing_gripper import PushingGripper
from .robotiq_three_finger_gripper import RobotiqThreeFingerGripper
from .panda_gripper import PandaGripper
from .panda_vices_gripper import PandaVICESGripper


def gripper_factory(name):
    """
    Genreator for grippers

    Creates a Gripper instance with the provided name.

    Args:
        name: the name of the gripper class

    Returns:
        gripper: Gripper instance

    Raises:
        XMLError: [description]
    """
    if name == "TwoFingerGripper":
        return TwoFingerGripper()
    if name == "LeftTwoFingerGripper":
        return LeftTwoFingerGripper()
    if name == "PR2Gripper":
        return PR2Gripper()
    if name == "RobotiqGripper":
        return RobotiqGripper()
    if name == "PushingGripper":
        return PushingGripper()
    if name == "RobotiqThreeFingerGripper":
        return RobotiqThreeFingerGripper()
    if name == "PandaGripper":
        return PandaGripper()
    if name == "PandaVICESGripper":
        return PandaVICESGripper()
    raise ValueError("Unkown gripper name {}".format(name))
