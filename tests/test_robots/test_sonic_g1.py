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
    SonicWholeBodyController, WBC_CONFIG)
from robosuite.utils.sonic.controller import MotorCommand


def _first_existing(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


GEAR_MODEL = _first_existing(
    "/home/amaddukuri/Projects/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml",
    "/home/ajay/code/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml",
)


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


def engine_pd_torques(engine, cmd, hands=None):
    """The SONIC per-motor PD law evaluated against the engine's CURRENT sim state,
    returned as ``{actuator_index: torque}`` clipped to per-motor effort:
    ``tau_i = clip(tau_ff_i + kp_i*(q*_i - q_i) + kd_i*(dq*_i - dq_i))``.
    This is the reference the SONIC_WBC part-controller dispatch must reproduce; it lives
    in test code because production drives the same law through JointPosition(Velocity)
    controllers, not this dict (read-only -- writes nothing)."""
    md = engine.sim.data._data if hasattr(engine.sim.data, "_data") else engine.sim.data
    out = {}
    q = md.qpos[engine.qpos_adr]
    dq = md.qvel[engine.qvel_adr]
    body = np.clip(cmd.tau + cmd.kp * (cmd.q - q) + cmd.kd * (cmd.dq - dq),
                   -engine.effort_limit, engine.effort_limit)
    for k, ci in enumerate(engine.ctrl_idx):
        out[int(ci)] = float(body[k])
    if hands is not None and engine.has_hands:
        lcmd, rcmd = hands
        for hc, (qa, va, ci), eff in ((lcmd, engine._lh, engine._lh_eff),
                                      (rcmd, engine._rh, engine._rh_eff)):
            ht = np.clip(hc.tau + hc.kp * (hc.q - md.qpos[qa]) + hc.kd * (hc.dq - md.qvel[va]),
                         -eff, eff)
            for k, c in enumerate(ci):
                out[int(c)] = float(ht[k])
    return out


def drive_engine_direct(engine, data):
    """Test-only direct PD substep (replaces the removed G1SonicController.apply): build
    obs + read the command, then write engine_pd_torques straight to ctrl. Used to drive
    the engine on a bare MjModel without robosuite's controller stack."""
    _obs, cmd, hands = engine.exchange()
    if cmd is None:
        data.ctrl[engine.ctrl_idx] = 0.0
        return
    for actidx, tau in engine_pd_torques(engine, cmd, hands).items():
        data.ctrl[actidx] = tau


def _action_from_gold(gold, idx=0):
    return np.concatenate([gold["cmd_q"][idx], gold["lh_cmd_q"][idx], gold["rh_cmd_q"][idx]])


def _gains_from_gold(gold, idx=0):
    return {
        "body": (np.asarray(gold["cmd_kp"][idx], dtype=float), np.asarray(gold["cmd_kd"][idx], dtype=float)),
        "lhand": (np.asarray(gold["lh_cmd_kp"][idx], dtype=float), np.asarray(gold["lh_cmd_kd"][idx], dtype=float)),
        "rhand": (np.asarray(gold["rh_cmd_kp"][idx], dtype=float), np.asarray(gold["rh_cmd_kd"][idx], dtype=float)),
    }


def _commands_from_action(action, gains):
    action = np.asarray(action, dtype=float)
    z_body = np.zeros(29)
    z_hand = np.zeros(7)
    cmd = MotorCommand(
        q=action[:29],
        dq=z_body.copy(),
        kp=gains["body"][0],
        kd=gains["body"][1],
        tau=z_body.copy(),
    )
    lh = MotorCommand(
        q=action[29:36],
        dq=z_hand.copy(),
        kp=gains["lhand"][0],
        kd=gains["lhand"][1],
        tau=z_hand.copy(),
    )
    rh = MotorCommand(
        q=action[36:43],
        dq=z_hand.copy(),
        kp=gains["rhand"][0],
        kd=gains["rhand"][1],
        tau=z_hand.copy(),
    )
    return cmd, (lh, rh)


# --- Test-only command sources (moved out of robosuite.utils.sonic.sources; production
# only ships DDSCommandSource). Both duck-type the source interface the engine needs:
# update(obs) + read() [+ read_hands()]. ---
class ReferenceMockSource:
    """Replays per-motor joint targets (motor order) as PD targets; advances one frame
    per read(), holds the last once exhausted."""
    def __init__(self, motor_targets, kp, kd):
        self.targets = np.asarray(motor_targets, dtype=np.float64)  # (T, n)
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.n = self.targets.shape[1]
        self.t = 0

    def update(self, obs):
        pass

    def read(self):
        idx = min(self.t, self.targets.shape[0] - 1)
        self.t += 1
        return MotorCommand(q=self.targets[idx], dq=np.zeros(self.n),
                            kp=self.kp, kd=self.kd, tau=np.zeros(self.n))


class ReplayCommandSource:
    """Deterministically replays a recorded per-step command stream (body + Dex3 hands)
    from a golden npz -- one frame per read(), holds the last once exhausted."""
    def __init__(self, gold):
        self.q, self.dq = gold["cmd_q"], gold["cmd_dq"]
        self.kp, self.kd, self.tau = gold["cmd_kp"], gold["cmd_kd"], gold["cmd_tau"]
        self.T = self.q.shape[0]
        self.has_hands = "lh_cmd_q" in gold
        if self.has_hands:
            self._lh = {f: gold[f"lh_cmd_{f}"] for f in ("q", "dq", "kp", "kd", "tau")}
            self._rh = {f: gold[f"rh_cmd_{f}"] for f in ("q", "dq", "kp", "kd", "tau")}
        self.t = self._cur = 0

    def update(self, obs):
        pass

    def read(self):
        i = self._cur = min(self.t, self.T - 1)
        self.t += 1
        return MotorCommand(self.q[i], self.dq[i], self.kp[i], self.kd[i], self.tau[i])

    def read_hands(self):
        if not self.has_hands:
            return None
        i = self._cur
        lc = MotorCommand(*[self._lh[f][i] for f in ("q", "dq", "kp", "kd", "tau")])
        rc = MotorCommand(*[self._rh[f][i] for f in ("q", "dq", "kp", "kd", "tau")])
        return lc, rc


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
        drive_engine_direct(sonic, data)
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

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(robosuite.__file__)),
                            "controllers", "config", "robots", "default_sonic_g1.json")
    cfg = load_composite_controller_config(controller=cfg_path)
    assert cfg["type"] == "SONIC_WBC"
    # control_freq=500 -> one sim substep (hence one PD dispatch) per env.step, so the
    # applied ctrl is computed at the same state we evaluate the engine reference at.
    env = robosuite.make("TwoArmLift", robots=["SonicG1Fixed"], controller_configs=cfg,
                         has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, control_freq=500)
    try:
        env.reset()
        cc = env.robots[0].composite_controller
        gains = {
            "body": (np.array(cc._cfg["MOTOR_KP"][:29], float), np.array(cc._cfg["MOTOR_KD"][:29], float)),
            "lhand": (np.full(7, 2.0), np.full(7, 0.1)),
            "rhand": (np.full(7, 2.0), np.full(7, 0.1)),
        }
        action = np.zeros(env.action_dim)
        cc.set_command_gains(gains)
        env.step(action)  # build the maps + part plan
        ref = None
        for _ in range(10):  # stop before the zero-target command collapses the fixed-base legs
            # The action command is constant, so the command we evaluate here is exactly the
            # command env.step dispatches. Compute the reference at the CURRENT state, before
            # env.step advances it.
            cmd, hands = _commands_from_action(action, gains)
            ref = engine_pd_torques(cc._maps, cmd, hands)  # {actuator_index: torque}
            env.step(action)                              # routed per-part PD dispatch
            idx = np.array(sorted(ref))                   # all 43 SONIC actuators
            applied = env.sim.data.ctrl[idx]
            engine = np.array([ref[i] for i in idx])
            assert np.allclose(applied, engine, atol=1e-9), \
                f"part-controller PD != engine PD (max {np.abs(applied - engine).max():.2e})"
        s = cc._maps  # both grippers actually exercised (14 hand actuators routed)
        assert s.has_hands and len(s._lh[2]) == 7 and len(s._rh[2]) == 7
        assert {int(c) for c in list(s._lh[2]) + list(s._rh[2])} <= set(ref)
        assert np.all(np.isfinite(env.sim.data.ctrl))
    finally:
        env.close()


# gold command streams live in the sonic_robosuite harness repo; override with $SONIC_GOLD_DIR.
GOLD_DIR = os.environ.get("SONIC_GOLD_DIR") or _first_existing(
    "/home/amaddukuri/Projects/sonic_robosuite/tests/gold",
    "/home/ajay/code/sonic_robosuite/tests/gold",
)


@pytest.mark.skipif(not os.path.exists(WBC_CONFIG), reason="gear_sonic config unavailable")
def test_sonic_g1_floating_base_in_env():
    """The free-floating SonicG1 keeps a TRUE top-level pelvis freejoint inside a real
    robosuite env (NullBase, not welded by the base machinery), and the SONIC obs reads
    the robot's own freejoint."""
    from robosuite.scripts.collect_sonic_g1_demos import SonicArenaEnv, SONIC_CFG, match_base_sim_physics
    env = SonicArenaEnv(robots=["SonicG1"], controller_configs=load_composite_controller_config(controller=SONIC_CFG),
                        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                        control_freq=500, hard_reset=False, ignore_done=True, horizon=10_000)
    try:
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
        env.step(np.zeros(env.action_dim))  # build maps
        assert cc._maps.free_qadr == int(m.jnt_qposadr[m.body_jntadr[pid]])
        for _ in range(20):
            env.step(np.zeros(env.action_dim))
        assert np.all(np.isfinite(env.sim.data.ctrl))
    finally:
        env.close()


@pytest.mark.skipif(not (os.path.exists(WBC_CONFIG) and os.path.isdir(GOLD_DIR)),
                    reason="gear_sonic config / gold command stream unavailable")
def test_sonic_data_collection_replay(tmp_path):
    """End-to-end: drive the native floating SonicG1 with the golden command stream and
    record a robosuite demo.hdf5 (states + per-step SONIC targets)."""
    import h5py
    from robosuite.scripts import collect_sonic_g1_demos as C
    from robosuite.wrappers import DataCollectionWrapper

    gold = dict(np.load(os.path.join(GOLD_DIR, "squat_001__A359.npz"), allow_pickle=True))
    env = None
    try:
        base = C.SonicArenaEnv(robots=["SonicG1"], controller_configs=load_composite_controller_config(controller=C.SONIC_CFG),
                               has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                               control_freq=500, hard_reset=False, ignore_done=True, horizon=10_000)
        tmp = str(tmp_path / "tmp")
        env = DataCollectionWrapper(base, tmp)
        env.reset()  # robot spawns standing via SonicG1.init_qpos
        C.match_base_sim_physics(base.sim.model._model)
        base.robots[0].composite_controller.set_command_gains(_gains_from_gold(gold))
        for t in range(60):
            env.step(_action_from_gold(gold, t))
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
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


@pytest.mark.skipif(not os.path.exists(WBC_CONFIG), reason="gear_sonic config unavailable")
def test_sonic_startup_band_is_object_safe():
    """The live-DDS startup hold is an elastic band -- a force on the PELVIS body only,
    NEVER a qpos write. Regression: the old freeze/fall-recovery pinned the WHOLE qpos
    vector, which would teleport any free task object (e.g. TwoArmLift's pot) back to a
    snapshot each step. Here we perturb the pot, step, and assert (a) the pot evolves
    freely (not snapped back), (b) the band applies a force to the pelvis, and (c)
    releasing the band zeros that force."""

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(robosuite.__file__)),
                            "controllers", "config", "robots", "default_sonic_g1.json")
    # floating SonicG1 in an env WITH a free task object (the pot)
    env = robosuite.make("TwoArmLift", robots=["SonicG1"],
                         controller_configs=load_composite_controller_config(controller=cfg_path),
                         has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        cc = env.robots[0].composite_controller
        m = env.sim.model._model
        md = env.sim.data._data
        env.step(np.zeros(env.action_dim))  # builds the engine + band; sets _pelvis_bid
        assert cc._pelvis_bid is not None, "floating base not detected"

        # locate a NON-robot freejoint (the pot) by its body name
        pelvis_bid = cc._pelvis_bid
        pot_qadr = next(
            int(m.jnt_qposadr[j]) for j in range(m.njnt)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and "pelvis" not in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]) or ""))

        # (a) perturb the pot, step, and confirm it was NOT snapped back to a snapshot
        before = np.array(md.qpos[pot_qadr:pot_qadr + 3])
        md.qpos[pot_qadr] += 0.15
        env.sim.forward()
        moved_to = float(md.qpos[pot_qadr])
        env.step(np.zeros(env.action_dim))
        after = float(md.qpos[pot_qadr])
        assert abs(after - before[0]) > 0.1, "pot was teleported back -- qpos clobbered!"
        assert abs(after - moved_to) < 0.05, "pot did not evolve from its perturbed pose"

        # (b) the band applies a force to the pelvis while enabled
        assert np.any(md.xfrc_applied[pelvis_bid] != 0.0), "band applied no force"

        # (c) releasing the band zeros that force
        cc.release_band()
        env.step(np.zeros(env.action_dim))
        assert np.all(md.xfrc_applied[pelvis_bid] == 0.0), "band force not cleared on release"
    finally:
        env.close()
