"""
Defines a string based method of initializing grippers
"""
from .panda_gripper import PandaGripper
from .pr2_gripper import PR2Gripper
from .rethink_gripper import RethinkGripper
from .robotiq_gripper import RobotiqGripper
from .robotiq_three_finger_gripper import RobotiqThreeFingerGripper


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
    if name == "RobotiqGripper":
        return RobotiqGripper(idn=idn)
    if name == "RobotiqThreeFingerGripper":
        return RobotiqThreeFingerGripper(idn=idn)
    if name == "PandaGripper":
        return PandaGripper(idn=idn)
    raise ValueError("Unknown gripper name: {}".format(name))
