"""
Defines a string based method of initializing grippers
"""
from .two_finger_gripper import TwoFingerGripper
from .pr2_gripper import PR2Gripper
from .robotiq_gripper import RobotiqGripper
from .pushing_gripper import PushingGripper
from .robotiq_three_finger_gripper import RobotiqThreeFingerGripper
from .panda_gripper import PandaGripper


def gripper_factory(name, idn=0):
    """
    Genreator for grippers

    Creates a Gripper instance with the provided name.

    Args:
        name: the name of the gripper class
        idn: idn (int or str): Number or some other unique identification string for this gripper instance

    Returns:
        gripper: Gripper instance

    Raises:
        XMLError: [description]
    """
    if name == "TwoFingerGripper":
        return TwoFingerGripper(idn=idn)
    if name == "PR2Gripper":
        return PR2Gripper(idn=idn)
    if name == "RobotiqGripper":
        return RobotiqGripper(idn=idn)
    if name == "PushingGripper":
        return PushingGripper(idn=idn)
    if name == "RobotiqThreeFingerGripper":
        return RobotiqThreeFingerGripper(idn=idn)
    if name == "PandaGripper":
        return PandaGripper(idn=idn)
    raise ValueError("Unknown gripper name {}".format(name))
