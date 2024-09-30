import json
import pathlib
from typing import Dict, Optional

import robosuite
from robosuite.controllers.parts.controller_factory import load_part_controller_config
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER


def validate_composite_controller_config(config: dict):
    # Check top-level keys
    required_keys = ["type", "body_parts_controller_configs"]
    for key in required_keys:
        if key not in config:
            ROBOSUITE_DEFAULT_LOGGER.error(f"Missing top-level key: {key}")
            raise ValueError


def load_composite_controller_config(controller: str = None, robot: str = None) -> Optional[Dict]:
    """
    Utility function that loads the desired composite controller and returns the loaded configuration as a dict

    TODO

    Args:
        controller: TODO.
        robot: Name of the robot to load the controller for. Currently only supports "GR1"

    Returns:
        dict: Controller configuration

    Raises:
        AssertionError: [Unknown default controller name]
        AssertionError: [No controller specified]
    """
    composite_controller_config = None
    # First check if default controller is not None; if it is not, load the appropriate controller
    if isinstance(controller, str) and controller.endswith(".json"):
        controller_fpath = controller
        ROBOSUITE_DEFAULT_LOGGER.info("Loading custom controller configuration from: {} ...".format(controller))
    else:
        if "GR1FloatingBody" in robot:
            robot_name = "gr1_floating_body"
        elif "GR1" in robot:
            robot_name = "gr1"
        elif "G1" in robot:
            robot_name = "g1"
        elif "H1" in robot:
            robot_name = "h1"
        else:
            robot_name = robot.lower()

        controller_fpath = (
            pathlib.Path(robosuite.__file__).parent / f"controllers/config/robots/default_{robot_name}.json"
        )

    # Attempt to load the controller
    try:
        with open(controller_fpath) as f:
            composite_controller_config = json.load(f)
    except FileNotFoundError:
        ROBOSUITE_DEFAULT_LOGGER.error(
            "Error opening controller filepath at: {}. " "Please check filepath and try again.".format(controller_fpath)
        )

    validate_composite_controller_config(composite_controller_config)
    # Load default controller configs for each specified body part
    body_parts_controller_configs = composite_controller_config.get("body_parts_controller_configs", {})
    composite_controller_config["body_parts"] = {}
    for part_name, part_config in body_parts_controller_configs.items():
        if part_name == "arms":
            for arm_name, arm_config in part_config.items():
                composite_controller_config["body_parts"][arm_name] = arm_config
        else:
            composite_controller_config["body_parts"][part_name] = part_config
    composite_controller_config.pop("body_parts_controller_configs")

    # Return the loaded controller
    return composite_controller_config
