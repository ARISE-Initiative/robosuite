"""
Test cross-process reproducibility of construction-time randomization.

When ``robosuite.make`` is called without an explicit ``seed``, the environment's
random number generator should be seeded from NumPy's global RNG, so that a
global ``np.random.seed(...)`` makes construction-time randomization (e.g.: the
cube size sampled in Lift) reproducible across processes. See issue #835.

Run this test with:
    $ python tests/test_environments/test_env_seed_reproducibility.py
"""
import numpy as np

import robosuite as suite


def make_lift(**kwargs):
    """
    Instantiates a lightweight, headless Lift environment.

    Args:
        **kwargs: Additional arguments to pass to the environment constructor

    Returns:
        MujocoEnv: The Lift environment instance
    """
    return suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,  # no on-screen renderer
        has_offscreen_renderer=False,  # no GL context required
        use_camera_obs=False,  # no camera observations
        **kwargs,
    )


def test_unseeded_env_follows_global_numpy_seed():
    """
    Two environments constructed with the same global NumPy seed (and no explicit
    ``seed`` argument) must be created identically, so that independent processes
    seeding NumPy globally get the same simulation.
    """
    np.random.seed(42)
    env1 = make_lift()
    np.random.seed(42)
    env2 = make_lift()
    xml1 = env1.sim.model.get_xml()
    xml2 = env2.sim.model.get_xml()
    env1.close()
    env2.close()
    assert xml1 == xml2, "Environments constructed under the same global NumPy seed differ!"


def test_explicit_seed_still_deterministic():
    """
    Passing an explicit ``seed`` to the environment must remain deterministic.
    """
    env1 = make_lift(seed=7)
    env2 = make_lift(seed=7)
    xml1 = env1.sim.model.get_xml()
    xml2 = env2.sim.model.get_xml()
    env1.close()
    env2.close()
    assert xml1 == xml2, "Environments constructed with the same explicit seed differ!"


def test_different_global_seeds_give_different_object_sizes():
    """
    Sanity check that different global NumPy seeds actually produce different
    construction-time randomization (i.e.: the seed is not being ignored).
    """
    np.random.seed(1)
    env1 = make_lift()
    np.random.seed(2)
    env2 = make_lift()
    xml1 = env1.sim.model.get_xml()
    xml2 = env2.sim.model.get_xml()
    env1.close()
    env2.close()
    assert xml1 != xml2, "Environments constructed under different global seeds are identical!"


if __name__ == "__main__":

    test_unseeded_env_follows_global_numpy_seed()
    test_explicit_seed_still_deterministic()
    test_different_global_seeds_give_different_object_sizes()
    print("All environment seed reproducibility tests passed successfully!")
