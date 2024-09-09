import json

import pathlib
from typing import Optional, Dict
import robosuite
from robosuite import load_controller_config


def load_composite_controller_config(custom_fpath: str = None, default_controller: str = None, robot: str = None) -> Optional[Dict]:
    """
    Utility function that loads the desired composite controller and returns the loaded configuration as a dict

    If @default_controller is specified, any value inputted to @custom_fpath is overridden and the default controller
    configuration is automatically loaded. See specific arg description below for available default controllers.

    Args:
        custom_fpath: Absolute filepath to the custom controller configuration .json file to be loaded
        default_controller: If specified, overrides @custom_fpath and loads a default configuration file for the
            specified controller.
        robot: Name of the robot to load the controller for. Currently only supports "GR1"

    Returns:
        dict: Controller configuration

    Raises:
        AssertionError: [Unknown default controller name]
        AssertionError: [No controller specified]
    """
    composite_controller_config = None
    # First check if default controller is not None; if it is not, load the appropriate controller
    if default_controller is not None:

        # Assert that requested default controller is in the available default controllers
        from robosuite.controllers.composite import ALL_CONTROLLERS

        assert (
            default_controller in ALL_CONTROLLERS
        ), "Error: Unknown default controller specified. Requested {}, " "available controllers: {}".format(
            default_controller, list(ALL_CONTROLLERS)
        )

        if "GR1" in robot:
            robot_name = "gr1"
        elif "Tiago" in robot:
            robot_name = "tiago"
        else:
            raise ValueError("Unknown robot name: {}".format(robot))

        # Store the default controller config fpath associated with the requested controller
        custom_fpath = pathlib.Path(robosuite.__file__).parent / f"controllers/config/composite/default_{default_controller.lower()}_{robot_name}.json"
    else:
        return None

    # Assert that the fpath to load the controller is not empty
    assert custom_fpath is not None, "Error: Either custom_fpath or default_controller must be specified!"

    # Attempt to load the controller
    try:
        with open(custom_fpath) as f:
            composite_controller_config = json.load(f)
    except FileNotFoundError:
        print("Error opening controller filepath at: {}. " "Please check filepath and try again.".format(custom_fpath))

    # Load default controller configs for each specified body part
    default_controller_configs_part_names = composite_controller_config.get("default_controller_configs_part_names", None)
    if default_controller_configs_part_names is not None:
        composite_controller_config["body_parts"] = {}
        for part_name in default_controller_configs_part_names:
            custom_fpath = pathlib.Path(robosuite.__file__).parent / f"controllers/config/default_{robot_name}.json"
            composite_controller_config["body_parts"][part_name] = load_controller_config(custom_fpath=custom_fpath)

    # Return the loaded controller
    return composite_controller_config
