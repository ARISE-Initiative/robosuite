import copy
import json
import os
import pathlib
from typing import Dict, Literal, Optional

import robosuite
from robosuite.controllers.composite.composite_controller import REGISTERED_COMPOSITE_CONTROLLERS_DICT
from robosuite.controllers.parts.controller_factory import load_part_controller_config
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER


def validate_composite_controller_config(config: dict):
    # Check top-level keys
    required_keys = ["type", "body_parts"]
    for key in required_keys:
        if key not in config:
            ROBOSUITE_DEFAULT_LOGGER.error(f"Missing top-level key: {key}")
            raise ValueError


def is_part_controller_config(config: Dict):
    """
    Checks if a controller config is a part config as a opposed to a composite
    controller config.

    Args:
        config (dict): Controller configuration

    Returns:
        bool: True if the config is in the for the arm-only, False otherwise
    """

    PART_CONTROLLER_TYPES = ["JOINT_VELOCITY", "JOINT_TORQUE", "JOINT_POSITION", "OSC_POSITION", "OSC_POSE", "IK_POSE"]
    if "body_parts" not in config and "type" in config:
        return config["type"] in PART_CONTROLLER_TYPES
    return False


def refactor_composite_controller_config(controller_config, robot_type, arms):
    """
    Checks if a controller config is in the format from robosuite versions <= 1.4.1.
    If this is the case, converts the controller config to the new composite controller
    config format in robosuite versions >= 1.5. If the robot has a default
    controller config use that and override the arms with the old controller config.
    If not just use the old controller config for arms.

    Args:
        old_controller_config (dict): Old controller config

    Returns:
        dict: New controller config
    """
    if not is_part_controller_config(controller_config):
        return controller_config

    config_dir = pathlib.Path(robosuite.__file__).parent / "controllers/config/robots/"
    name = robot_type.lower()
    configs = os.listdir(config_dir)
    if f"default_{name}.json" in configs:
        new_controller_config = load_composite_controller_config(robot=name)
    else:
        new_controller_config = {}
        new_controller_config["type"] = "BASIC"
        new_controller_config["body_parts"] = {}

    for arm in arms:
        new_controller_config["body_parts"][arm] = copy.deepcopy(controller_config)
        new_controller_config["body_parts"][arm]["gripper"] = {"type": "GRIP"}
    return new_controller_config


def load_composite_controller_config(controller: Optional[str] = None, robot: Optional[str] = None) -> Optional[Dict]:
    """
    Utility function that loads the desired composite controller and returns the loaded configuration as a dict

    Args:
        controller: Name or path of the controller to load.
            If None, robot must be specified and we load the robot's default controller in controllers/config/robots/.
            If specified, must be a valid controller name or path to a controller config file.
        robot: Name of the robot to load the controller for.

    Returns:
        dict: Controller configuration

    Raises:
        FileNotFoundError: If the controller file is not found
    """
    # Determine the controller file path
    if controller is None:
        assert robot is not None, "If controller is None, robot must be specified."
        # Load robot's controller
        robot_name = _get_robot_name(robot)
        controller_fpath = (
            pathlib.Path(robosuite.__file__).parent / f"controllers/config/robots/default_{robot_name}.json"
        )
        if not os.path.exists(controller_fpath):
            controller_fpath = (
                pathlib.Path(robosuite.__file__).parent / "controllers/config/default/composite/basic.json"
            )
    elif isinstance(controller, str):
        if controller.endswith(".json"):
            # Use the specified path directly
            controller_fpath = controller
        else:
            assert (
                controller in REGISTERED_COMPOSITE_CONTROLLERS_DICT
            ), f"Controller {controller} not found in COMPOSITE_CONTROLLERS_DICT"
            # Load from robosuite/controllers/config/default/composite/
            controller_name = controller.lower()
            controller_fpath = (
                pathlib.Path(robosuite.__file__).parent / f"controllers/config/default/composite/{controller_name}.json"
            )
    else:
        raise ValueError("Controller must be None or a string.")

    # Attempt to load the controller
    try:
        with open(controller_fpath) as f:
            composite_controller_config = json.load(f)
        ROBOSUITE_DEFAULT_LOGGER.info(f"Loading controller configuration from: {controller_fpath}")
    except FileNotFoundError:
        ROBOSUITE_DEFAULT_LOGGER.error(
            f"Error opening controller filepath at: {controller_fpath}. Please check filepath and try again."
        )
        raise

    validate_composite_controller_config(composite_controller_config)
    body_parts_controller_configs = composite_controller_config.pop("body_parts", {})
    composite_controller_config["body_parts"] = {}
    for part_name, part_config in body_parts_controller_configs.items():
        if part_name == "arms":
            for arm_name, arm_config in part_config.items():
                composite_controller_config["body_parts"][arm_name] = arm_config
        else:
            composite_controller_config["body_parts"][part_name] = part_config

    return composite_controller_config


def _get_robot_name(robot: str) -> str:
    """Helper function to get the standardized robot name."""
    if "GR1FloatingBody" in robot:
        return "gr1_floating_body"
    elif "GR1FixedLowerBody" in robot:
        return "gr1_fixed_lower_body"
    elif "GR1" in robot:
        return "gr1"
    elif "G1" in robot:
        return "g1"
    elif "H1" in robot:
        return "h1"
    elif "PandaDex" in robot:
        return "panda_dex"
    else:
        return robot.lower()
