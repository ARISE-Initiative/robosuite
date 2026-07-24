"""
Test that DomainRandomizationWrapper survives a hard reset.

When an environment uses the "mujoco" renderer, a hard reset frees the old MjSim
and builds a new one. The wrapper's modders keep a reference to the freed sim, so
saving the default domain must rebind them to the new sim first. Otherwise the
second reset raises `AttributeError: 'MjSim' object has no attribute 'model'`.

See https://github.com/ARISE-Initiative/robosuite/issues/426.
"""
import numpy as np

import robosuite as suite
from robosuite.wrappers import DomainRandomizationWrapper


def test_domain_randomization_hard_reset():

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        renderer="mujoco",
    )
    # randomize_color needs mujoco==3.1.1, so leave it off here
    env = DomainRandomizationWrapper(env, randomize_color=False)

    # the first reset builds the initial sim; each later hard reset frees it and
    # builds a new one, which is what used to break the modders
    for _ in range(3):
        env.reset()
        env.step(np.zeros(env.action_dim))

    env.close()


if __name__ == "__main__":

    test_domain_randomization_hard_reset()
