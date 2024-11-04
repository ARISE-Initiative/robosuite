import argparse
import time
from typing import Dict, List, Union

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.utils.robot_composition_utils import create_composite_robot


def create_and_test_env(
    env: str,
    robots: Union[str, List[str]],
    controller_config: Dict,
    headless: bool = False,
    max_fr: int = None,
):

    config = {
        "env_name": env,
        "robots": robots,
        "controller_configs": controller_config,
    }

    env = suite.make(
        **config,
        has_renderer=not headless,
        has_offscreen_renderer=headless,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env.reset()
    low, high = env.action_spec
    low = np.clip(low, -1, 1)
    high = np.clip(high, -1, 1)

    # Runs a few steps of the simulation as a sanity check
    for i in range(100):
        start = time.time()

        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--robot", type=str, required=True)
    parser.add_argument("--base", type=str, default=None)
    parser.add_argument("--grippers", nargs="+", type=str, default=["PandaGripper"])
    parser.add_argument("--env", type=str, default="Lift")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--max_fr", default=20, type=int, help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time."
    )

    args = parser.parse_args()

    if args.robot not in ROBOT_CLASS_MAPPING:
        raise ValueError(f"Robot {args.robot} not found in ROBOT_CLASS_MAPPING \n" f"{ROBOT_CLASS_MAPPING.keys()}")

    name = f"Custom{args.robot}"
    create_composite_robot(name, base=args.base, robot=args.robot, grippers=args.grippers)
    controller_config = load_composite_controller_config(controller="BASIC", robot=name)

    create_and_test_env(
        env="Lift", robots=name, controller_config=controller_config, headless=args.headless, max_fr=args.max_fr
    )
