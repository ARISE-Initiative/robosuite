"""Tests for the SonicG1 robot, its Dex3 grippers and the SONIC_WBC controller.

Apart from the registration test they need gear_sonic installed (the controller reads its config)
and skip without it. The recorded-torque test also needs $SONIC_RECORDED_RUNS_DIR: runs of the
SONIC C++ controller recorded with the standalone sonic_robosuite harness (its tests/gold
directory), one .npz per motion holding the joint states, SONIC's commands and the torques that
harness applied. The harness and the recordings are not published.
"""
import os
import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.models import MujocoWorldBase
from robosuite.models.arenas import EmptyArena
from robosuite.models.grippers.sonic_dex3_gripper import SonicDex3LeftGripper, SonicDex3RightGripper
from robosuite.models.robots.manipulators.sonic_g1_robot import SonicG1
from robosuite.scripts.collect_sonic_g1_demos import SONIC_CFG, match_base_sim_physics
from robosuite.utils.sonic.bridge import BODY, LEFT_HAND, MOTOR_KD, MOTOR_KP, RIGHT_HAND

RECORDED_RUNS_DIR = os.environ.get("SONIC_RECORDED_RUNS_DIR", "")
RECORDED_MOTIONS = ["squat_001__A359", "walking_quip_360_R_002__A428", "neutral_kick_R_001__A543",
                    "forward_lunge_R_001__A359_M", "tired_one_leg_jumping_R_001__A359",
                    "dance_in_da_party_001__A464", "macarena_001__A545"]


def _make_env(env_name, control_freq):
    pytest.importorskip("gear_sonic")
    return robosuite.make(env_name, robots=["SonicG1"],
                          controller_configs=load_composite_controller_config(controller=SONIC_CFG),
                          has_renderer=False, has_offscreen_renderer=False,
                          use_camera_obs=False, control_freq=control_freq)


def test_registration():
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.models.grippers import GRIPPER_MAPPING
    from robosuite.controllers.composite.composite_controller import (
        REGISTERED_COMPOSITE_CONTROLLERS_DICT)
    assert "SonicG1" in ROBOT_CLASS_MAPPING
    assert "SonicDex3LeftGripper" in GRIPPER_MAPPING
    assert "SonicDex3RightGripper" in GRIPPER_MAPPING
    assert "SONIC_WBC" in REGISTERED_COMPOSITE_CONTROLLERS_DICT


def test_assembles():
    """SonicG1 + Dex3 grippers assemble to 43 actuators and, compared with gear_sonic's source
    model, to the same total mass."""
    robot = SonicG1(idn=0)
    robot.add_gripper(SonicDex3RightGripper(idn="0_right"), robot.eef_name["right"])
    robot.add_gripper(SonicDex3LeftGripper(idn="0_left"), robot.eef_name["left"])
    world = MujocoWorldBase()
    world.merge(EmptyArena())
    world.merge(robot)
    model = world.get_model(mode="mujoco")
    assert model.nu == 43

    gear_sonic = pytest.importorskip("gear_sonic")
    source = mujoco.MjModel.from_xml_path(os.path.join(
        os.path.dirname(gear_sonic.__file__), "data", "robot_model", "model_data", "g1", "g1_29dof_with_hand.xml"))
    assert source.nu == 43
    assert abs(sum(model.body_mass) - sum(source.body_mass)) < 1e-3


def test_motor_map_follows_sonic_motor_order():
    leg = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
    arm = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    finger = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
    expected = ([f"robot0_l_leg_{joint}_joint" for joint in leg] + [f"robot0_r_leg_{joint}_joint" for joint in leg]
                + [f"robot0_torso_waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
                + [f"robot0_l_{joint}_joint" for joint in arm] + [f"robot0_r_{joint}_joint" for joint in arm]
                + [f"gripper0_left_left_hand_{joint}_joint" for joint in finger]
                + [f"gripper0_right_right_hand_{joint}_joint" for joint in finger])
    env = _make_env("Lift", control_freq=20)
    try:
        env.reset()
        motor_map = env.robots[0].composite_controller.motor_map
        assert motor_map.joint_names == expected
        assert np.array_equal(motor_map.effort_limit[15:29], [25, 25, 25, 25, 25, 5, 5] * 2)  # both arms
        assert np.array_equal(motor_map.effort_limit[LEFT_HAND], motor_map.effort_limit[RIGHT_HAND])
    finally:
        env.close()


@pytest.mark.parametrize("target_offset, reaches_effort_limit", [(0.02, False), (2.0, True)])
def test_sonic_wbc_applies_sonic_pd_law(target_offset, reaches_effort_limit):
    """The torque SONIC_WBC applies through robosuite's per-part controllers equals SONIC's PD law
    evaluated directly on the motor arrays: tau = clip(kp * (target - q) - kd * dq, effort). Every
    motor gets a different target. Near the joints no motor reaches its effort limit, so a routing
    swap between any two motors shows; far away, body and hand motors are clipped."""
    # control_freq=500 -> one sim substep per env.step, so ctrl was computed at the pre-step state
    env = _make_env("TwoArmLift", control_freq=500)
    try:
        env.reset()
        motor_map = env.robots[0].composite_controller.motor_map
        data = env.sim.data._data
        action = data.qpos[motor_map.qpos_adr] + np.linspace(-target_offset, target_offset, env.action_dim)
        clipped = np.zeros(env.action_dim, dtype=bool)
        for _ in range(10):
            torque = MOTOR_KP * (action - data.qpos[motor_map.qpos_adr]) - MOTOR_KD * data.qvel[motor_map.qvel_adr]
            expected = np.clip(torque, -motor_map.effort_limit, motor_map.effort_limit)
            clipped |= np.abs(torque) > motor_map.effort_limit
            env.step(action)
            assert np.allclose(data.ctrl[motor_map.actuator_ids], expected, atol=1e-9)
        if reaches_effort_limit:
            assert clipped[BODY].any() and clipped[LEFT_HAND].any() and clipped[RIGHT_HAND].any()
        else:
            assert not clipped.any()
    finally:
        env.close()


def test_match_base_sim_physics_sets_sonic_physics_options():
    env = _make_env("Lift", control_freq=500)
    try:
        env.reset()
        model = env.sim.model._model
        match_base_sim_physics(model)
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
        assert model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
        assert model.opt.impratio == 1.0 and model.opt.density == 0.0 and model.opt.viscosity == 0.0
        floor_geoms = [geom for geom in range(model.ngeom) if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE]
        assert floor_geoms and all(model.geom_friction[geom, 0] == 1.0 for geom in floor_geoms)
    finally:
        env.close()


def test_sonic_g1_floating_base_in_env():
    """SonicG1 keeps a top-level pelvis freejoint inside a robosuite env, and the motor map
    resolves that joint as the robot's base."""
    env = _make_env("Lift", control_freq=500)
    try:
        env.reset()
        model = env.sim.model._model
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_pelvis")
        assert model.body_parentid[pelvis] == 0  # child of the world
        assert model.body_jntnum[pelvis] == 1
        assert model.jnt_type[model.body_jntadr[pelvis]] == int(mujoco.mjtJoint.mjJNT_FREE)
        motor_map = env.robots[0].composite_controller.motor_map
        assert motor_map.pelvis_body_id == pelvis
        assert motor_map.base_qpos_adr == int(model.jnt_qposadr[model.body_jntadr[pelvis]])
        for _ in range(20):
            env.step(np.zeros(env.action_dim))
        assert np.all(np.isfinite(env.sim.data.qpos))
    finally:
        env.close()


@pytest.mark.skipif(not os.path.isdir(RECORDED_RUNS_DIR),
                    reason="set SONIC_RECORDED_RUNS_DIR to the recorded runs of the SONIC controller")
@pytest.mark.parametrize("motion", RECORDED_MOTIONS)
def test_sonic_wbc_reproduces_recorded_torques(motion):
    """At the joint states and SONIC commands of a recorded run (every 16th frame), SONIC_WBC
    applies the torques the recording harness applied, for the body and both hands: same gains
    and PD law. Body motors reach their effort limit in the one-leg jumping and the macarena."""
    recorded = dict(np.load(os.path.join(RECORDED_RUNS_DIR, motion + ".npz"), allow_pickle=True))
    for motors, prefix in ((BODY, "cmd"), (LEFT_HAND, "lh_cmd"), (RIGHT_HAND, "rh_cmd")):
        assert np.array_equal(recorded[f"{prefix}_kp"][0], MOTOR_KP[motors])
        assert np.array_equal(recorded[f"{prefix}_kd"][0], MOTOR_KD[motors])
    env = _make_env("Lift", control_freq=20)
    try:
        env.reset()
        robot = env.robots[0]
        motor_map = robot.composite_controller.motor_map
        data = env.sim.data._data
        # robosuite's part controllers read the joint state cached at reset on their first run
        robot.control(np.ones(env.action_dim), policy_step=True)
        worst_error = 0.0
        for frame in range(0, len(recorded["cmd_q"]), 16):
            data.qpos[motor_map.qpos_adr] = np.concatenate(
                [recorded["body_q"][frame], recorded["left_hand_q"][frame], recorded["right_hand_q"][frame]])
            data.qvel[motor_map.qvel_adr] = np.concatenate(
                [recorded["body_dq"][frame], recorded["left_hand_dq"][frame], recorded["right_hand_dq"][frame]])
            env.sim.forward()
            targets = np.concatenate(
                [recorded["cmd_q"][frame], recorded["lh_cmd_q"][frame], recorded["rh_cmd_q"][frame]])
            robot.control(targets, policy_step=True)
            expected = np.concatenate(
                [recorded["tau_body"][frame], recorded["tau_lh"][frame], recorded["tau_rh"][frame]])
            worst_error = max(worst_error, float(np.abs(data.ctrl[motor_map.actuator_ids] - expected).max()))
        assert worst_error < 1e-6, f"{motion}: applied torque differs from the recorded one by {worst_error:.2e}"
    finally:
        env.close()


class _FakeUnitreeDds:
    """Stands in for gear_sonic's UnitreeSdk2Bridge: holds the latest commands without opening DDS."""

    def __init__(self, sonic_config):
        self.low_cmd_lock, self.left_hand_cmd_lock, self.right_hand_cmd_lock = (threading.Lock() for _ in range(3))
        self.published = []
        self.reset()

    def reset(self):
        self.low_cmd_received = self.left_hand_cmd_received = self.right_hand_cmd_received = False

    def PublishLowState(self, obs):
        self.published.append(obs)

    def receive(self, targets, kp, kd):
        """Store one body and two hand commands holding the 43 targets and gains, in motor order."""
        motor_cmds = [SimpleNamespace(q=target, kp=motor_kp, kd=motor_kd)
                      for target, motor_kp, motor_kd in zip(targets, kp, kd)]
        self.low_cmd = SimpleNamespace(motor_cmd=motor_cmds[BODY])
        self.left_hand_cmd = SimpleNamespace(motor_cmd=motor_cmds[LEFT_HAND])
        self.right_hand_cmd = SimpleNamespace(motor_cmd=motor_cmds[RIGHT_HAND])
        self.low_cmd_received = self.left_hand_cmd_received = self.right_hand_cmd_received = True


def test_bridge_returns_targets_only_while_sonic_commands(monkeypatch):
    """SonicRobosuiteBridge.get_sonic_action against a fake DDS layer: it always publishes the robot
    state, and returns SONIC's 43 targets only once body and hand commands with SONIC's gains arrive."""
    pytest.importorskip("unitree_sdk2py")
    gear_sonic_dds = pytest.importorskip("gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge")
    import unitree_sdk2py.core.channel as unitree_channel
    from robosuite.utils.sonic import bridge as sonic_bridge

    monkeypatch.setattr(gear_sonic_dds, "UnitreeSdk2Bridge", _FakeUnitreeDds)
    monkeypatch.setattr(unitree_channel, "ChannelFactoryInitialize", lambda *args: None)
    monkeypatch.setattr(sonic_bridge, "_dds_initialized", False)
    env = _make_env("Lift", control_freq=20)
    try:
        env.reset()
        bridge = sonic_bridge.SonicRobosuiteBridge(env.robots[0].composite_controller.sonic_config)
        dds = bridge.unitree_dds
        targets = np.linspace(-0.3, 0.3, env.action_dim)

        assert bridge.get_sonic_action(env) is None  # nothing received yet
        assert dds.published[-1]["body_q"].shape == (29,) and dds.published[-1]["left_hand_q"].shape == (7,)

        dds.receive(targets, MOTOR_KP, MOTOR_KD)
        assert np.array_equal(bridge.get_sonic_action(env), targets)

        dds.right_hand_cmd_received = False  # a hand command is still missing
        assert bridge.get_sonic_action(env) is None

        damping_kp, damping_kd = MOTOR_KP.copy(), MOTOR_KD.copy()
        damping_kp[BODY], damping_kd[BODY] = 0.0, 8.0  # SONIC's damping-only stop command
        dds.receive(targets, damping_kp, damping_kd)
        assert bridge.get_sonic_action(env) is None

        for changed_motor in (BODY.start, LEFT_HAND.start, RIGHT_HAND.start):  # gains changed on the SONIC side
            changed_kp = MOTOR_KP.copy()
            changed_kp[changed_motor] *= 2
            dds.receive(targets, changed_kp, MOTOR_KD)
            with pytest.raises(RuntimeError):
                bridge.get_sonic_action(env)

        dds.receive(targets, MOTOR_KP, MOTOR_KD)
        bridge.reset()
        assert bridge.get_sonic_action(env) is None
    finally:
        env.close()


def test_sonic_startup_band_holds_reference_yaw():
    """The startup band is referenced to the spawn pose, so it does not yaw the robot back to
    world identity when an env spawns it facing a fixture."""
    from scipy.spatial.transform import Rotation

    env = _make_env("TwoArmLift", control_freq=20)
    try:
        env.reset()
        controller = env.robots[0].composite_controller
        motor_map = controller.motor_map
        data = env.sim.data._data
        pelvis, qpos_adr, qvel_adr = motor_map.pelvis_body_id, motor_map.base_qpos_adr, motor_map.base_qvel_adr

        def set_pelvis_rotation(rotation):
            quat = rotation.as_quat()
            data.qpos[qpos_adr + 3:qpos_adr + 7] = [quat[3], quat[0], quat[1], quat[2]]
            data.qvel[qvel_adr:qvel_adr + 6] = 0.0
            env.sim.forward()

        spawn_rotation = Rotation.from_euler("z", np.pi / 2.0)
        set_pelvis_rotation(spawn_rotation)
        env.step(np.zeros(env.action_dim))  # anchors the band at this pose
        set_pelvis_rotation(spawn_rotation)
        controller._apply_band()
        assert abs(data.xfrc_applied[pelvis, 5]) < 1e-6

        set_pelvis_rotation(spawn_rotation * Rotation.from_euler("x", 0.1))
        controller._apply_band()
        assert np.linalg.norm(data.xfrc_applied[pelvis, 3:5]) > 10.0
        assert abs(data.xfrc_applied[pelvis, 5]) < 1e-6
    finally:
        env.close()


def test_sonic_startup_band_is_object_safe():
    """The startup band is a force on the pelvis only, never a qpos write: a perturbed task object
    (TwoArmLift's pot) evolves freely, and releasing the band zeros the force. With no SONIC
    command (zero action) the controller applies no torque."""
    env = _make_env("TwoArmLift", control_freq=20)
    try:
        env.reset()
        controller = env.robots[0].composite_controller
        motor_map = controller.motor_map
        model = env.sim.model._model
        data = env.sim.data._data
        pelvis = motor_map.pelvis_body_id
        pot_qpos_adr = next(
            int(model.jnt_qposadr[joint]) for joint in range(model.njnt)
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_bodyid[joint] != pelvis)

        env.step(np.zeros(env.action_dim))
        before = float(data.qpos[pot_qpos_adr])
        data.qpos[pot_qpos_adr] += 0.15
        env.sim.forward()
        moved_to = float(data.qpos[pot_qpos_adr])
        env.step(np.zeros(env.action_dim))
        after = float(data.qpos[pot_qpos_adr])
        assert abs(after - before) > 0.1, "pot was teleported back -- qpos clobbered!"
        assert abs(after - moved_to) < 0.05, "pot did not evolve from its perturbed pose"

        assert np.any(data.xfrc_applied[pelvis] != 0.0), "band applied no force"
        assert not data.ctrl[motor_map.actuator_ids].any(), "zero action applied torque"

        controller.release_band()
        env.step(np.zeros(env.action_dim))
        assert np.all(data.xfrc_applied[pelvis] == 0.0), "band force not cleared on release"
    finally:
        env.close()
