"""
Checks that the object-to-eef relative observables in NutAssembly, PickPlace and ToolHang are computed from the
current object pose, i.e. the absolute object sensors update before the relative ones read them from obs_cache.
"""

import numpy as np
import pytest

import robosuite as suite
import robosuite.utils.transform_utils as T

ENVS = [
    ("NutAssemblySquare", ["SquareNut"]),
    ("PickPlaceCan", ["Can"]),
    ("ToolHang", ["base", "frame", "tool"]),
]


def _expected_rel_pos(obs, obj, eef_prefix):
    obj_pose = T.pose2mat((obs[f"{obj}_pos"], obs[f"{obj}_quat"]))
    eef_pose = T.pose2mat((obs[f"{eef_prefix}eef_pos"], obs[f"{eef_prefix}eef_quat"]))
    rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, T.pose_inv(eef_pose))
    return T.mat2pose(rel_pose)[0]


@pytest.mark.parametrize("env_name,objs", ENVS)
def test_rel_obj_eef_pos_uses_current_object_pose(env_name, objs):
    env = suite.make(
        env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    try:
        obs = env.reset()
        for step in range(3):
            for obj in objs:
                key = f"{obj}_to_robot0_eef_pos"
                assert key in obs
                np.testing.assert_allclose(
                    obs[key], _expected_rel_pos(obs, obj, "robot0_"), atol=1e-5, err_msg=f"{key} at step {step}"
                )
                # quaternions q and -q encode the same rotation; compare rotation matrices
                quat_key = f"{obj}_to_robot0_eef_quat"
                expected_rotation = T.quat2mat(obs["robot0_eef_quat"]).T @ T.quat2mat(obs[f"{obj}_quat"])
                np.testing.assert_allclose(
                    np.linalg.norm(obs[quat_key]), 1.0, atol=1e-5, err_msg=f"{quat_key} norm at step {step}"
                )
                np.testing.assert_allclose(
                    T.quat2mat(obs[quat_key]), expected_rotation, atol=1e-5, err_msg=f"{quat_key} at step {step}"
                )
            obs, _, _, _ = env.step(np.random.uniform(-1, 1, env.action_dim))
    finally:
        env.close()
