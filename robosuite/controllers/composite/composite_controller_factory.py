import json
import pathlib
from typing import Dict, Literal, Optional

import robosuite
from robosuite.controllers.composite.composite_controller import REGISTERED_COMPOSITE_CONTROLLERS_DICT
from robosuite.controllers.parts.controller_factory import load_part_controller_config
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER


def validate_composite_controller_config(config: dict):
    # Check top-level keys
    required_keys = ["type", "body_parts_controller_configs"]
    for key in required_keys:
        if key not in config:
            ROBOSUITE_DEFAULT_LOGGER.error(f"Missing top-level key: {key}")
            raise ValueError


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
        controller_fpath = pathlib.Path(robosuite.__file__).parent / f"controllers/config/robots/default_{robot_name}.json"
    elif isinstance(controller, str):
        if controller.endswith(".json"):
            # Use the specified path directly
            controller_fpath = controller
        else:
            assert controller in REGISTERED_COMPOSITE_CONTROLLERS_DICT, f"Controller {controller} not found in REGISTERED_COMPOSITE_CONTROLLERS_DICT"
            # Load from robosuite/controllers/config/default/composite/
            controller_name = controller.lower()
            controller_fpath = pathlib.Path(robosuite.__file__).parent / f"controllers/config/default/composite/{controller_name}.json"
    else:
        raise ValueError("Controller must be None or a string.")

    # Attempt to load the controller
    try:
        with open(controller_fpath) as f:
            composite_controller_config = json.load(f)
        ROBOSUITE_DEFAULT_LOGGER.info(f"Loading controller configuration from: {controller_fpath}")
    except FileNotFoundError:
        ROBOSUITE_DEFAULT_LOGGER.error(f"Error opening controller filepath at: {controller_fpath}. Please check filepath and try again.")
        raise

    validate_composite_controller_config(composite_controller_config)
    body_parts_controller_configs = composite_controller_config.pop("body_parts_controller_configs", {})
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
    elif "GR1" in robot:
        return "gr1"
    elif "G1" in robot:
        return "g1"
    elif "H1" in robot:
        return "h1"
    else:
        return robot.lower()
