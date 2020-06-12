from .gripper_model import GripperModel
from .gripper_factory import gripper_factory
from .gripper_tester import GripperTester

from .panda_gripper import PandaGripper
from .pr2_gripper import PR2Gripper
from .rethink_gripper import RethinkGripper
from .robotiq_85_gripper import Robotiq85Gripper
from .robotiq_three_finger_gripper import RobotiqThreeFingerGripper
from .panda_gripper import PandaGripper
from .jaco_three_finger_gripper import JacoThreeFingerGripper
from .robotiq_140_gripper import Robotiq140Gripper
from .wiping_gripper import WipingGripper
from .null_gripper import NullGripper


GRIPPER_MAPPING = {
    "RethinkGripper": RethinkGripper,
    "PR2Gripper": PR2Gripper,
    "Robotiq85Gripper": Robotiq85Gripper,
    "RobotiqThreeFingerGripper": RobotiqThreeFingerGripper,
    "PandaGripper": PandaGripper,
    "JacoThreeFingerGripper": JacoThreeFingerGripper,
    "Robotiq140Gripper": Robotiq140Gripper,
    "WipingGripper": WipingGripper,
    None: NullGripper,
}

ALL_GRIPPERS = GRIPPER_MAPPING.keys()
