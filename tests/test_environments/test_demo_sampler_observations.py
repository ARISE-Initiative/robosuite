"""Check observations after restoring genuine demonstration simulation states."""

import h5py
import numpy as np
import pytest

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DemoSamplerWrapper


@pytest.fixture
def demonstration(tmp_path):
    env = robosuite.make(
        "Lift",
        robots=["Panda"],
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=37,
    )
    try:
        cached_position = env.reset()["cube_pos"].copy()
        joint = env.cube.joints[0]
        qpos = env.sim.data.get_joint_qpos(joint).copy()
        qpos[0] += 0.08
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
        state = np.array(env.sim.get_state().flatten())
        position = env.sim.data.body_xpos[env.cube_body_id].copy()
        assert not np.allclose(cached_position, position)
        # Direct state writes must not be mistaken for a fresh sensor sample.
        np.testing.assert_array_equal(env._get_observations()["cube_pos"], cached_position)

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "task.xml").write_text(env.sim.model.get_xml())
        with h5py.File(tmp_path / "demo.hdf5", "w") as demo:
            data = demo.create_group("data")
            data.attrs["env"] = "Lift"
            episode = data.create_group("demo_0")
            episode.attrs["model_file"] = "task.xml"
            episode.create_dataset("states", data=state[None])
        yield env, tmp_path, state, position
    finally:
        env.close()


@pytest.mark.parametrize("scheme", ["uniform", "forward", "reverse"])
@pytest.mark.parametrize("need_xml", [False, True])
def test_sampled_reset_returns_current_observations(demonstration, scheme, need_xml):
    env, demo_path, state, position = demonstration
    sampler = DemoSamplerWrapper(
        env,
        str(demo_path),
        need_xml=need_xml,
        sampling_schemes=(scheme,),
        scheme_ratios=(1.0,),
    )
    try:
        observations = sampler.reset()
        np.testing.assert_allclose(observations["cube_pos"], position, rtol=0, atol=1e-12)
        np.testing.assert_allclose(observations["object-state"][:3], position, rtol=0, atol=1e-12)
        np.testing.assert_allclose(env.sim.get_state().flatten(), state, rtol=0, atol=1e-12)
    finally:
        sampler.demo_file.close()


@pytest.mark.parametrize("need_xml", [False, True])
def test_random_reset_returns_current_observations(demonstration, need_xml):
    env, demo_path, _, _ = demonstration
    sampler = DemoSamplerWrapper(
        env,
        str(demo_path),
        need_xml=need_xml,
        sampling_schemes=("random",),
        scheme_ratios=(1.0,),
    )
    try:
        observations = sampler.reset()
        position = env.sim.data.body_xpos[env.cube_body_id].copy()
        np.testing.assert_allclose(observations["cube_pos"], position, rtol=0, atol=1e-12)
        np.testing.assert_allclose(observations["object-state"][:3], position, rtol=0, atol=1e-12)
        assert env.timestep == 0
    finally:
        sampler.demo_file.close()
