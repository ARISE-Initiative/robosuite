import time

from robosuite.utils.input_utils import *
from robosuite.controllers import load_composite_controller_config

MAX_FR = 25  # max frame rate for running simluation

if __name__ == "__main__":

    controller_config = load_composite_controller_config(
        robot="SpotWithArmFloating",
    )

    # initialize the task
    env = suite.make(
        env_name="Lift",
        robots=["SpotWithArmFloating"],
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    print(env.action_spec)

    # Get action limits
    low, high = env.action_spec

    print(len(low))

    # do visualization
    for i in range(10000):
        start = time.time()

        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()

        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
