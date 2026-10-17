"""
Regression tests for PickPlace task success checking (issue #830).

An object sitting inside its correct target bin must NOT be counted as placed
while the robot gripper is still grasping it. Prior to the fix, "released" was
proxied by gripper-to-object-center distance (``r_reach < 0.6``, i.e. gripper
more than ~4.24 cm away), which tall objects (e.g. Milk, Cereal) grasped near
the top also satisfy while still held, inflating ``objects_in_bins`` (and
therefore sparse reward and task success).

The tests below build a held-in-bin state headlessly by:
  1. servoing the (open) gripper above the target bin with the default OSC
     position servo,
  2. teleporting the object upright between the fingers via its free joint,
  3. closing the finger joints slightly so both fingerpads statically touch
     the object (contacts are computed by ``sim.forward()``, no stepping).

Run with:
    python -m pytest tests/test_environments/test_pick_place_success.py -q
"""

import numpy as np

import robosuite as suite
from robosuite.utils import transform_utils as T

# In-bin offset from the milk's nominal target bin placement, keeping the
# teleported object strictly inside the bin bounds checked by not_in_bin()
# (x/y inside the bin quadrant, z inside (bin2_pos[2], bin2_pos[2] + 0.1)).
BIN_OFFSET = (0.02, 0.02)
OBJ_Z = 0.82

# Vertical distance between the gripper site and the object center when
# grasping near the top of the carton: larger than the 4.24 cm release-proxy
# threshold so the old buggy condition (r_reach < 0.6) is satisfied, but small
# enough that the fingerpads still overlap the upper part of the milk carton.
GRASP_DROP = 0.07

# Height of the gripper above the object center for the released-object case.
HOVER_DROP = 0.23

# Finger joint angles: squeeze for the static grasp (both fingerpads touching
# the carton sides), fully open for the released case (Panda finger joint
# range is [0, 0.04] / [-0.04, 0]).
SQUEEZE_Q = 0.012
OPEN_Q = 0.04

# Convergence tolerance (m) for the gripper servo.
SERVO_TOL = 0.004


def _make_env():
    """Create a headless single-object (Milk) PickPlace environment."""
    return suite.make(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        single_object_mode=2,
        object_type="milk",
        control_freq=20,
        horizon=1000,
        ignore_done=True,
        hard_reset=False,
        seed=42,
    )


def _servo_eef_to(env, obs, target_pos, max_steps=200):
    """Servo the gripper site to @target_pos with the default OSC controller."""
    target_pos = np.asarray(target_pos, dtype=np.float64)
    arm = env.robots[0].arms[0]
    for _ in range(max_steps):
        eef_pos = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]])
        err_world = target_pos - eef_pos
        # The default OSC takes deltas in the robot base frame; rotate the
        # world-frame error into the base frame. Actions are normalized to
        # [-1, 1] and scaled by the controller's output_max (~5 cm/step).
        root_body = env.robots[0].robot_model.root_body
        base_quat = np.array(env.sim.model.body_quat[env.sim.model.body_name2id(root_body)])
        base_rot_mat = T.quat2mat(T.convert_quat(base_quat, to="xyzw"))
        err_base = base_rot_mat.T @ err_world
        dpos = np.clip(2.0 * err_base, -1.0, 1.0)
        action = np.concatenate([dpos, [0.0, 0.0, 0.0, 1.0]])  # keep gripper open
        obs, _, done, _ = env.step(action)
        if np.linalg.norm(err_world) < SERVO_TOL:
            break
        if done:
            break
    residual = np.linalg.norm(target_pos - np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]]))
    assert residual < 0.01, "gripper servo did not converge (residual {:.4f} m)".format(residual)
    return obs


def _teleport_object_to(env, pos):
    """Teleport the active object upright at @pos via its free joint."""
    sim = env.sim
    obj = env.objects[env.object_id]
    joint = obj.joints[0]
    qpos = sim.data.get_joint_qpos(joint).copy()
    qpos[:3] = pos
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # upright, (w, x, y, z)
    sim.data.set_joint_qpos(joint, qpos)
    jid = sim.model.joint_name2id(joint)
    dofadr = sim.model.jnt_dofadr[jid]
    sim.data.qvel[dofadr : dofadr + 6] = 0.0
    sim.forward()
    return np.array(sim.data.body_xpos[env.obj_body_id[obj.name]])


def _set_finger_qpos(env, q):
    """Set both gripper finger joints to +-q and refresh derived quantities."""
    sim = env.sim
    finger_joints = sorted(env.robots[0].gripper[env.robots[0].arms[0]].joints)
    assert len(finger_joints) == 2, "expected two finger joints, got {}".format(finger_joints)
    sim.data.set_joint_qpos(finger_joints[0], q)
    sim.data.set_joint_qpos(finger_joints[1], -q)
    sim.forward()


def _object_state(env):
    """Return (obj_pos, gripper-object distance) for the active object."""
    obj = env.objects[env.object_id]
    obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
    arm = env.robots[0].arms[0]
    dist = np.linalg.norm(np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]]) - obj_pos)
    return obj_pos, dist


def test_held_in_bin_object_is_not_counted_as_placed():
    """An object grasped inside its target bin must not count as placed (issue #830)."""
    env = _make_env()
    try:
        obs = env.reset()

        # Servo the open gripper above the milk's target bin, then place the
        # object upright between the fingers and squeeze.
        target = env.target_bin_placements[env.object_id].copy()
        in_bin_pos = np.array([target[0] + BIN_OFFSET[0], target[1] + BIN_OFFSET[1], OBJ_Z])
        grasp_pos = in_bin_pos + np.array([0.0, 0.0, GRASP_DROP])
        obs = _servo_eef_to(env, obs, grasp_pos)
        obj_pos = _teleport_object_to(env, in_bin_pos)
        _set_finger_qpos(env, SQUEEZE_Q)

        # Sanity: the state must be exactly the buggy one reported in the
        # issue -- object inside the bin bounds, gripper beyond the old
        # release-proxy distance, and yet actively grasping the object.
        in_bin = not env.not_in_bin(obj_pos, env.object_id)
        _, dist = _object_state(env)
        grasping = env._check_grasp(
            gripper=env.robots[0].gripper,
            object_geoms=env.objects[env.object_id].contact_geoms,
        )
        assert in_bin, "test setup failed: object not inside target bin bounds"
        assert dist > 0.0424, "test setup failed: gripper closer than release proxy ({:.4f} m)".format(dist)
        assert grasping, "test setup failed: gripper is not grasping the object"

        # Regression: the held object must not be counted as placed.
        success = env._check_success()
        assert (
            env.objects_in_bins[env.object_id] == 0
        ), "held-in-bin object counted as placed (objects_in_bins={})".format(env.objects_in_bins)
        assert not success
    finally:
        env.close()


def test_released_in_bin_object_is_still_counted_as_placed():
    """An object inside its target bin with the gripper away still counts as placed."""
    env = _make_env()
    try:
        obs = env.reset()

        # Hover the gripper well above the bin and drop the object in, with
        # fingers open and far from the object.
        target = env.target_bin_placements[env.object_id].copy()
        in_bin_pos = np.array([target[0] + BIN_OFFSET[0], target[1] + BIN_OFFSET[1], OBJ_Z])
        hover_pos = in_bin_pos + np.array([0.0, 0.0, HOVER_DROP])
        obs = _servo_eef_to(env, obs, hover_pos)
        obj_pos = _teleport_object_to(env, in_bin_pos)
        _set_finger_qpos(env, OPEN_Q)

        in_bin = not env.not_in_bin(obj_pos, env.object_id)
        _, dist = _object_state(env)
        grasping = env._check_grasp(
            gripper=env.robots[0].gripper,
            object_geoms=env.objects[env.object_id].contact_geoms,
        )
        assert in_bin, "test setup failed: object not inside target bin bounds"
        assert dist > 0.0424, "test setup failed: gripper closer than release proxy ({:.4f} m)".format(dist)
        assert not grasping, "test setup failed: gripper is unexpectedly grasping the object"

        success = env._check_success()
        assert (
            env.objects_in_bins[env.object_id] == 1
        ), "released-in-bin object not counted as placed (objects_in_bins={})".format(env.objects_in_bins)
        assert success
    finally:
        env.close()
