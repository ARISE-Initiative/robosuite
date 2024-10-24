from typing import List, Optional, Tuple, Union

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS, RobotModel
from robosuite.robots import register_robot_class
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
from robosuite.utils.robot_utils import check_bimanual

BASE_TARGET_MAPPING = {
    "RethinkMount": "FixedBaseRobot",
    "RethinkMinimalMount": "FixedBaseRobot",
    "NullMount": "FixedBaseRobot",
    "OmronMobileBase": "WheeledRobot",
    "NullMobileBase": "WheeledRobot",
    "NoActuationBase": "LeggedRobot",
    "Spot": "LeggedRobot",
    "SpotFloating": "LeggedRobot",
}


def get_target_type(base) -> str:
    """
    Returns the target type of the robot
    """
    return BASE_TARGET_MAPPING[base]


def create_composite_robot(
    name: str, robot: str, base: Optional[str] = None, grippers: Optional[Union[str, List[str], Tuple[str]]] = None
) -> RobotModel:
    """
    Factory function to create a composite robot
    """

    bimanual_robot = check_bimanual(robot)
    grippers = grippers if type(grippers) == list or type(grippers) == tuple else [grippers]

    # perform checks and issues warning if necessary
    if bimanual_robot and len(grippers) == 1:
        grippers = grippers + grippers
    if not bimanual_robot and len(grippers) == 2:
        ROBOSUITE_DEFAULT_LOGGER.warning(
            f"Grippers {grippers} supplied for single gripper robot.\
                                           Using gripper {grippers[0]}."
        )
    if not base:
        base = REGISTERED_ROBOTS[robot]().default_base
    if robot in ["Tiago", "GR1"] and base:
        ROBOSUITE_DEFAULT_LOGGER.warning(f"Defined custom base when using {robot} robot. Ignoring base.")
        if robot in ["Tiago"]:
            base = "NullMobileBase"
        elif robot == "GR1":
            base = "NoActuationBase"

    target_type = get_target_type(base)

    class_dict = {
        "default_base": property(lambda self: base),
        "default_arms": property(lambda self: {"right": robot}),
        "default_gripper": property(
            lambda self: {"right": grippers[0], "left": grippers[1]} if bimanual_robot else {"right": grippers[0]}
        ),
    }

    CustomCompositeRobotClass = type(name, (REGISTERED_ROBOTS[robot],), class_dict)
    register_robot_class(target_type)(CustomCompositeRobotClass)

    return CustomCompositeRobotClass
