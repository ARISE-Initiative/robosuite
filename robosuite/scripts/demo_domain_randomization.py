import numpy as np
import robosuite as suite
from robosuite.wrappers import DRWrapper
from robosuite.environments.sawyer_lift import SawyerLift

from PIL import Image

if __name__ == "__main__":

    # initialize the task
    env = suite.make(
        "SawyerLift",
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=100,
        camera_name="agentview",
    )
    env = DRWrapper(env)
    env.reset()
    env.viewer.set_camera(camera_id=0)

    for i in range(10000):
        action = np.random.randn(env.dof)
        obs, reward, done, _ = env.step(action)
        env.render()
