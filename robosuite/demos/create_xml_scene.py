import argparse
import numpy as np
import mujoco
import robosuite as suite

from robosuite.robots import MobileRobot
from robosuite.utils.input_utils import *
import time

MAX_FR = 25  # max frame rate for running simulation

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an XML scene with robosuite and save the result to a file."
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Path to output XML file. If not provided, a default filename is generated based on the environment and chosen robots."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "single-robot":
            options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True, exclude_single_arm=True)
        else:
            options["robots"] = []

            # Have user choose two robots
            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=False, use_humanoids=True))
    # If a humanoid environment has been chosen, choose humanoid robots
    elif "Humanoid" in options["env_name"]:
        options["robots"] = choose_robots(use_humanoids=True)
    else:
        options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True)

    # Initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    env.reset()
    env.viewer.set_camera(camera_id=0)
    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    # Run a simulation step and render the scene
    action = np.random.randn(*env.action_spec[0].shape)
    obs, reward, done, _ = env.step(action)
    env.render()

    # input("Press anything to continue...")
    time.sleep(1)

    # Reload the object positions to save the updated scene state.
    # NOTE: This is a hacky way to save the object placements. The saved scene might differ from what is rendered.
    if hasattr(env, "placement_initializer"):
        object_placements = env.placement_initializer.sample()
        for obj_pos, obj_quat, obj in object_placements.values():
            # Get the body id for the object's root body
            body_id = env.sim.model.body_name2id(obj.root_body)
            env.sim.model._model.body_pos[body_id] = np.array(obj_pos)
            env.sim.model._model.body_quat[body_id] = np.array(obj_quat)

    # Determine the output filename
    if args.output:
        filename = args.output
    else:
        # Default filename generated from environment and robots selection
        filename = "scenes/scene_{}_{}.xml".format(options["env_name"], options["robots"])

    mujoco.mj_saveLastXML(filename, env.sim.model._model)

    print("Done. Scene saved to:", filename)
