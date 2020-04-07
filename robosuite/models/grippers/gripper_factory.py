"""
Defines a string based method of initializing grippers
"""
from .sawyer_gripper import SawyerGripper
from .pr2_gripper import PR2Gripper
from .robotiq_gripper import RobotiqGripper
from .pushing_sawyer_gripper import PushingSawyerGripper
from .robotiq_three_finger_gripper import RobotiqThreeFingerGripper
from .panda_gripper import PandaGripper


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
    if name == "SawyerGripper":
        return SawyerGripper(idn=idn)
    if name == "PR2Gripper":
        return PR2Gripper(idn=idn)
    if name == "RobotiqGripper":
        return RobotiqGripper(idn=idn)
    if name == "PushingSawyerGripper":
        return PushingSawyerGripper(idn=idn)
    if name == "RobotiqThreeFingerGripper":
        return RobotiqThreeFingerGripper(idn=idn)
    if name == "PandaGripper":
        return PandaGripper(idn=idn)
    raise ValueError("Unknown gripper name: {}".format(name))
