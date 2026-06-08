from .gripper_model import GripperModel
from .gripper_factory import gripper_factory
from .gripper_tester import GripperTester

from .panda_gripper import PandaGripper
from .rethink_gripper import RethinkGripper
from .robotiq_85_gripper import Robotiq85Gripper
from .robotiq_three_finger_gripper import RobotiqThreeFingerGripper, RobotiqThreeFingerDexterousGripper
from .panda_gripper import PandaGripper
from .jaco_three_finger_gripper import JacoThreeFingerGripper, JacoThreeFingerDexterousGripper
from .robotiq_140_gripper import Robotiq140Gripper
from .wiping_gripper import WipingGripper
from .bd_gripper import BDGripper
from .null_gripper import NullGripper
from .inspire_hands import InspireLeftHand, InspireRightHand
from .fourier_hands import FourierLeftHand, FourierRightHand
from .xarm7_gripper import XArm7Gripper
from .suction_gripper import SuctionGripper

GRIPPER_MAPPING = {
    "RethinkGripper": RethinkGripper,
    "PandaGripper": PandaGripper,
    "JacoThreeFingerGripper": JacoThreeFingerGripper,
    "JacoThreeFingerDexterousGripper": JacoThreeFingerDexterousGripper,
    "WipingGripper": WipingGripper,
    "Robotiq85Gripper": Robotiq85Gripper,
    "Robotiq140Gripper": Robotiq140Gripper,
    "RobotiqThreeFingerGripper": RobotiqThreeFingerGripper,
    "RobotiqThreeFingerDexterousGripper": RobotiqThreeFingerDexterousGripper,
    "BDGripper": BDGripper,
    "InspireLeftHand": InspireLeftHand,
    "InspireRightHand": InspireRightHand,
    "FourierLeftHand": FourierLeftHand,
    "FourierRightHand": FourierRightHand,
    "XArm7Gripper": XArm7Gripper,
    "SuctionGripper": SuctionGripper,
    None: NullGripper,
}

ALL_GRIPPERS = GRIPPER_MAPPING.keys()


def register_gripper(target_class):
    GRIPPER_MAPPING[target_class.__name__] = target_class
    return target_class


# Imported after register_gripper is defined so the @register_gripper decorator
# resolves (these grippers self-register into GRIPPER_MAPPING).
from .sonic_dex3_gripper import SonicDex3LeftGripper, SonicDex3RightGripper  # noqa: E402,F401
