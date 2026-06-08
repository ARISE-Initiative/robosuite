"""Tests for the native SonicG1 robot + Dex3 grippers + SonicWholeBodyController.

Covers:
  - registration (SonicG1 / SonicG1Fixed / Dex3 grippers / SONIC_WBC)
  - assembly: robosuite RobotModel + Dex3 grippers reassemble to the SAME physics as
    the source model (43 DOF, mass preserved)
  - OSC via robosuite.make: SonicG1Fixed in a task env, arms controlled by OSC_POSE
    (the right EEF moves under an OSC pose command)  [self-contained]
  - SONIC_WBC: the SONIC composite controller drives the assembled robot (right arm
    gets its real 25/5 Nm effort, ctrl finite)  [needs external gear_sonic config]
  - floating base: the free-floating SonicG1 keeps a top-level pelvis freejoint in a
    real robosuite env (NullBase) and SONIC reads the robot's base, not task clutter
  - data collection: the native SonicG1 + SONIC_WBC records a robosuite demo.hdf5
    (states + per-step SONIC targets)  [needs gold command stream]
"""
import os

import mujoco
import numpy as np
import pytest

import robosuite  # noqa: F401  registers SonicG1[/Fixed] + Dex3 grippers + SONIC_WBC
from robosuite.controllers import load_composite_controller_config
from robosuite.models import MujocoWorldBase
from robosuite.models.arenas import EmptyArena
from robosuite.models.grippers.sonic_dex3_gripper import SonicDex3LeftGripper, SonicDex3RightGripper
from robosuite.models.robots.manipulators.sonic_g1_robot import SonicG1, SonicG1Fixed
from robosuite.controllers.composite.sonic_whole_body_controller import (
    SonicWholeBodyController, set_sonic_source_factory, WBC_CONFIG)

GEAR_MODEL = ("/home/ajay/code/GR00T-WholeBodyControl/gear_sonic/"
              "data/robot_model/model_data/g1/g1_29dof_with_hand.xml")


def _assemble():
    """SonicG1Fixed + Dex3 grippers merged into an EmptyArena (robosuite machinery)."""
    m = SonicG1Fixed(idn=0)
    m.add_gripper(SonicDex3RightGripper(idn="0_right"), m.eef_name["right"])
    m.add_gripper(SonicDex3LeftGripper(idn="0_left"), m.eef_name["left"])
    world = MujocoWorldBase()
    world.merge(EmptyArena())
    world.merge(m)
    return world.get_model(mode="mujoco")


class _Sim:
    """Minimal robosuite-sim duck type for G1SonicController."""
    def __init__(self, m, d):
        self.model = type("M", (), {"_model": m})()
        self.data = type("D", (), {"_data": d})()


def test_registration():
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.models.grippers import GRIPPER_MAPPING
    from robosuite.controllers.composite.composite_controller import (
        REGISTERED_COMPOSITE_CONTROLLERS_DICT)
    assert "SonicG1" in ROBOT_CLASS_MAPPING and "SonicG1Fixed" in ROBOT_CLASS_MAPPING
    assert "SonicDex3LeftGripper" in GRIPPER_MAPPING
    assert "SonicDex3RightGripper" in GRIPPER_MAPPING
    assert "SONIC_WBC" in REGISTERED_COMPOSITE_CONTROLLERS_DICT


def test_assembles():
    model = _assemble()
    assert model.nu == 43  # 29 body + 7 + 7 Dex3
    if os.path.exists(GEAR_MODEL):
        orig = mujoco.MjModel.from_xml_path(GEAR_MODEL)
        assert model.nu == orig.nu
        rmass = sum(model.body_mass[b] for b in range(model.nbody)
                    if any(k in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                           for k in ["hip", "knee", "ankle", "waist", "torso", "shoulder", "elbow",
                                     "wrist", "hand", "pelvis", "link", "gripper", "eef", "base"]))
        assert abs(rmass - sum(orig.body_mass)) < 1e-3  # physics preserved


def test_osc_robosuite_make():
    """SonicG1Fixed loads into a task env and OSC moves the right arm's EEF."""
    cfg = load_composite_controller_config(controller="BASIC")  # OSC_POSE arms
    env = robosuite.make("TwoArmLift", robots=["SonicG1Fixed"], controller_configs=cfg,
                         has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        low, _ = env.action_spec
        grip = env.robots[0].gripper["right"].important_sites["grip_site"]
        eef0 = env.sim.data.get_site_xpos(grip).copy()
        action = np.zeros_like(low)
        action[0] = 0.6  # +x pose delta on the right arm
        for _ in range(50):
            env.step(action)
        eef1 = env.sim.data.get_site_xpos(grip).copy()
        assert np.all(np.isfinite(env.sim.data.ctrl))
        assert np.linalg.norm(eef1 - eef0) > 0.02, "OSC did not move the right EEF"
    finally:
        env.close()


@pytest.mark.skipif(not os.path.exists(WBC_CONFIG), reason="gear_sonic config unavailable")
def test_engine_pd_and_effort_on_assembled_model():
    """The SONIC engine (G1SonicController) drives the assembled robot directly via its
    PD law (bypass path): the right arm gets its real 25/5 Nm effort and the write
    stays finite/bounded."""
    import yaml
    from robosuite.utils.sonic.controller import G1SonicController
    from robosuite.utils.sonic.sources import ReferenceMockSource
    with open(WBC_CONFIG) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = _assemble()
    model.opt.timestep = 0.002
    model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
    model.opt.cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    model.opt.impratio = 1.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    src = ReferenceMockSource(np.zeros((1, 29)),
                              np.array(cfg["MOTOR_KP"][:29], float),
                              np.array(cfg["MOTOR_KD"][:29], float))
    sonic = G1SonicController(_Sim(model, data), src, cfg)
    assert sonic.num_motors == 29 and sonic.has_hands
    # right arm must get its real effort (25/5 Nm), not the left-hand 0.7 Nm
    ra = [i for i, n in enumerate(sonic.motor_joint_names) if "r_shoulder" in n or "r_elbow" in n]
    assert ra and all(sonic.effort_limit[i] >= 25.0 for i in ra)

    for _ in range(200):
        sonic.apply()
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.ctrl)) and np.all(np.isfinite(data.qpos))
    assert np.abs(data.ctrl).max() < 200.0


@pytest.mark.skipif(not os.path.exists(WBC_CONFIG), reason="gear_sonic config unavailable")
def test_sonic_wbc_robosuite_make_matches_engine():
    """SONIC_WBC drives SonicG1Fixed inside a robosuite.make env by routing the DDS
    command to per-part JointPositionControllers (PD law from live state). The applied
    ctrl matches the SONIC engine's own PD torque (compute_torques) on every actuator
    -- legs/torso/arms AND both grippers -- confirming the controller computes the law
    (no left/right swap, no gravcomp/scaling drift, hands included)."""
    from robosuite.utils.sonic.controller import MotorCommand
    from robosuite.utils.sonic.sources import ReferenceMockSource

    class _MockWithHands(ReferenceMockSource):
        """Mock that also streams a (zero-target) Dex3 hand PD command so the grippers
        are driven and verified, not skipped."""
        def __init__(self, targets, kp, kd, hkp, hkd):
            super().__init__(targets, kp, kd)
            self._hkp, self._hkd = hkp, hkd

        def read_hands(self):
            z = np.zeros(7)
            mk = lambda: MotorCommand(q=z.copy(), dq=z.copy(), kp=self._hkp, kd=self._hkd, tau=z.copy())
            return mk(), mk()

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(robosuite.__file__)),
                            "controllers", "config", "robots", "default_sonic_g1.json")
    set_sonic_source_factory(lambda c: _MockWithHands(
        np.zeros((1, 29)), np.array(c["MOTOR_KP"][:29], float), np.array(c["MOTOR_KD"][:29], float),
        np.full(7, 2.0), np.full(7, 0.1)))
    try:
        cfg = load_composite_controller_config(controller=cfg_path)
        assert cfg["type"] == "SONIC_WBC"
        env = robosuite.make("TwoArmLift", robots=["SonicG1Fixed"], controller_configs=cfg,
                             has_renderer=False, has_offscreen_renderer=False,
                             use_camera_obs=False, control_freq=20)
        try:
            env.reset()
            cc = env.robots[0].composite_controller
            cc.verify = True  # stash the engine reference torque each step
            for _ in range(10):  # stop before the zero-target mock collapses the fixed-base legs
                env.step(np.zeros(env.action_dim))
                idx = np.array(sorted(cc.ref_ctrl))  # all 43 SONIC actuators
                applied = env.sim.data.ctrl[idx]
                engine = np.array([cc.ref_ctrl[i] for i in idx])
                assert np.allclose(applied, engine, atol=1e-9), \
                    f"part-controller PD != engine PD (max {np.abs(applied - engine).max():.2e})"
            s = cc._sonic  # both grippers actually exercised (14 hand actuators routed)
            assert s.has_hands and len(s._lh[2]) == 7 and len(s._rh[2]) == 7
            assert {int(c) for c in list(s._lh[2]) + list(s._rh[2])} <= set(cc.ref_ctrl)
            assert np.all(np.isfinite(env.sim.data.ctrl))
        finally:
            env.close()
    finally:
        set_sonic_source_factory(None)


GOLD_DIR = "/home/ajay/code/sonic_robosuite/tests/gold"


def _mock_factory():
    """Source factory: zero-target body PD + (zero) Dex3 hand PD, so grippers run too."""
    from robosuite.utils.sonic.controller import MotorCommand
    from robosuite.utils.sonic.sources import ReferenceMockSource

    class _MockWithHands(ReferenceMockSource):
        def read_hands(self):
            z = np.zeros(7)
            mk = lambda: MotorCommand(q=z.copy(), dq=z.copy(), kp=np.full(7, 2.0), kd=np.full(7, 0.1), tau=z.copy())
            return mk(), mk()

    return lambda c: _MockWithHands(np.zeros((1, 29)), np.array(c["MOTOR_KP"][:29], float),
                                    np.array(c["MOTOR_KD"][:29], float))


@pytest.mark.skipif(not os.path.exists(WBC_CONFIG), reason="gear_sonic config unavailable")
def test_sonic_g1_floating_base_in_env():
    """The free-floating SonicG1 keeps a TRUE top-level pelvis freejoint inside a real
    robosuite env (NullBase, not welded by the base machinery), and the SONIC obs reads
    the robot's own freejoint."""
    from robosuite.scripts.collect_sonic_g1_demos import SonicArenaEnv, SONIC_CFG, match_base_sim_physics
    from robosuite.utils.sonic.controller import G1SonicController
    set_sonic_source_factory(_mock_factory())
    try:
        env = SonicArenaEnv(robots=["SonicG1"], controller_configs=load_composite_controller_config(controller=SONIC_CFG),
                            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                            control_freq=500, hard_reset=False, ignore_done=True, horizon=10_000)
        env.reset()
        match_base_sim_physics(env.sim.model._model)
        m = env.sim.model._model
        pid = next(b for b in range(m.nbody) if "pelvis" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""))
        # pelvis is top-level (child of world) and carries a free joint -> truly floating
        assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[pid]) == "world"
        assert m.body_jntnum[pid] == 1
        assert m.jnt_type[m.body_jntadr[pid]] == int(mujoco.mjtJoint.mjJNT_FREE)
        # SONIC controller resolves the ROBOT's freejoint for its base obs
        cc = env.robots[0].composite_controller
        cc._sonic = G1SonicController(cc.sim, cc._src, cc._cfg)
        assert cc._sonic.free_qadr == int(m.jnt_qposadr[m.body_jntadr[pid]])
        for _ in range(20):
            env.step(np.zeros(env.action_dim))
        assert np.all(np.isfinite(env.sim.data.ctrl))
        env.close()
    finally:
        set_sonic_source_factory(None)


@pytest.mark.skipif(not (os.path.exists(WBC_CONFIG) and os.path.isdir(GOLD_DIR)),
                    reason="gear_sonic config / gold command stream unavailable")
def test_sonic_data_collection_replay(tmp_path):
    """End-to-end: drive the native floating SonicG1 with the golden command stream and
    record a robosuite demo.hdf5 (states + per-step SONIC targets)."""
    import h5py
    from robosuite.scripts import collect_sonic_g1_demos as C
    from robosuite.wrappers import DataCollectionWrapper

    factory, gold = C.make_source_factory("replay", "squat_001__A359")
    set_sonic_source_factory(factory)
    try:
        base = C.SonicArenaEnv(robots=["SonicG1"], controller_configs=load_composite_controller_config(controller=C.SONIC_CFG),
                               has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                               control_freq=500, hard_reset=False, ignore_done=True, horizon=10_000)
        tmp = str(tmp_path / "tmp")
        env = DataCollectionWrapper(base, tmp)
        env.reset()
        C.match_base_sim_physics(base.sim.model._model)
        C.set_init_pose(base, gold)
        for _ in range(60):
            env.step(C.sonic_action(base))
        env.close()
        out = str(tmp_path / "out")
        C.gather_to_hdf5(tmp, out, "SonicArenaEnv", {"type": "SONIC_WBC", "robot": "SonicG1"})
        with h5py.File(os.path.join(out, "demo.hdf5"), "r") as f:
            demos = list(f["data"].keys())
            assert len(demos) >= 1
            s = f["data"][demos[0]]["states"][:]
            a = f["data"][demos[0]]["actions"][:]
            assert s.shape[0] == a.shape[0] and s.shape[0] > 0
            assert a.shape[1] == base.action_dim
            assert not np.all(a == 0)  # SONIC targets recorded
            assert f["data"][demos[0]].attrs["model_file"]  # replayable model
    finally:
        set_sonic_source_factory(None)
