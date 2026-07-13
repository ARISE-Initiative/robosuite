"""G1SonicController: whole-body torque bridge for robosuite.

Mirrors what ``gear_sonic``'s ``base_sim`` does on the MuJoCo side, but inside a
robosuite env: every control substep it
  1. builds the proprioceptive obs dict from ``sim.data`` (== base_sim.prepare_obs),
  2. hands it to a *command source* (publishes lowstate over DDS to the C++
     SONIC controller, or a local mock),
  3. reads back the per-motor command (q*, dq*, kp, kd, tau_ff),
  4. applies SONIC's per-motor PD law and writes torques to ``sim.data.ctrl``.

The PD law is ported verbatim from ``base_sim.compute_body_torques``:
    tau_i = tau_ff_i + kp_i*(q*_i - q_i) + kd_i*(dq*_i - dq_i)
with gains coming from the command stream (no mass-matrix decoupling, no gravity
compensation) — i.e. the real Unitree motor behavior, NOT robosuite's
JointPositionController.
"""

import mujoco
import numpy as np

# Body-part keywords used by base_sim to identify the 29 actuated body joints.
_BODY_JOINT_KEYWORDS = ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]

# Dex3 hand joints, 7 per side, in the order the C++ sends rt/dex3/*/cmd
# (== gear_sonic joint_utils.G1_HAND_JOINTS).
_LEFT_HAND = ["left_hand_index_0_joint", "left_hand_index_1_joint",
              "left_hand_middle_0_joint", "left_hand_middle_1_joint",
              "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint"]
_RIGHT_HAND = ["right_hand_index_0_joint", "right_hand_index_1_joint",
               "right_hand_middle_0_joint", "right_hand_middle_1_joint",
               "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"]

# Canonical joint order of config["motor_effort_limit_list"] (model_data 43-DOF
# ACTUATOR order: legs, waist, L-arm, L-hand, R-arm, R-hand). We map effort limits
# by NAME (not actuator index) so it's correct regardless of how the model is
# assembled -- the integrated model (legs,waist,L-arm,L-hand,R-arm,R-hand) AND the
# native robosuite robot (body 29 then split-out grippers) both resolve right.
_EFFORT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "left_hand_thumb_0", "left_hand_thumb_1", "left_hand_thumb_2",
    "left_hand_middle_0", "left_hand_middle_1", "left_hand_index_0", "left_hand_index_1",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    "right_hand_thumb_0", "right_hand_thumb_1", "right_hand_thumb_2",
    "right_hand_middle_0", "right_hand_middle_1", "right_hand_index_0", "right_hand_index_1",
]


def _effort_by_name(eff_list):
    """name->effort from the canonical 43-DOF list, longest-key-first for matching."""
    d = dict(zip(_EFFORT_NAMES, eff_list))
    keys = sorted(d, key=len, reverse=True)
    return d, keys


def _lookup_effort(joint_name, eff_d, eff_keys, default=5.0):
    for k in eff_keys:  # longest first so e.g. left_hip_pitch beats no shorter key
        if k in joint_name:
            return eff_d[k]
    return default


class MotorCommand:
    """Per-motor command (length = num_motors), all in Unitree motor order."""

    __slots__ = ("q", "dq", "kp", "kd", "tau")

    def __init__(self, q, dq, kp, kd, tau):
        self.q, self.dq, self.kp, self.kd, self.tau = q, dq, kp, kd, tau


class G1SonicController:
    """Reads sim state, exchanges it with a command source, and writes torques.

    A command source needs ``update(obs)`` (consume latest obs, e.g. publish lowstate)
    and ``read()`` (return a MotorCommand or None); optionally ``read_hands()``. The only
    production source is DDSCommandSource; tests inject their own."""

    def __init__(self, sim, command_source, config: dict):
        self.sim = sim
        self.src = command_source
        self.last_obs = None
        m = sim.model._model if hasattr(sim.model, "_model") else sim.model

        # --- motor index maps (actuator order == Unitree motor order) ---
        self.num_motors = int(config["NUM_MOTORS"])
        self.qpos_adr, self.qvel_adr, self.ctrl_idx, self.motor_joint_names = [], [], [], []
        for a in range(m.nu):
            jid = int(m.actuator_trnid[a, 0])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if any(k in name for k in _BODY_JOINT_KEYWORDS):
                self.qpos_adr.append(int(m.jnt_qposadr[jid]))
                self.qvel_adr.append(int(m.jnt_dofadr[jid]))
                self.ctrl_idx.append(a)
                self.motor_joint_names.append(name)
        assert len(self.ctrl_idx) == self.num_motors, (
            f"expected {self.num_motors} body actuators, found {len(self.ctrl_idx)}"
        )
        self.qpos_adr = np.array(self.qpos_adr)
        self.qvel_adr = np.array(self.qvel_adr)
        self.ctrl_idx = np.array(self.ctrl_idx)

        # --- free joint + torso bookkeeping (for obs); prefix-tolerant so it works
        # on bare model_data names AND robosuite's "robot0_"-prefixed names ---
        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if fj < 0:  # native robosuite robot: locate the free joint by type
            frees = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
            # prefer the ROBOT's own base (pelvis) over any task-object freejoints
            # (e.g. a TwoArmLift pot) so build_obs publishes the robot base, not clutter
            fj = next((j for j in frees if "pelvis" in
                       (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]) or "")),
                      frees[0] if frees else -1)
        self.free_qadr = int(m.jnt_qposadr[fj]) if fj >= 0 else None
        self.free_vadr = int(m.jnt_dofadr[fj]) if fj >= 0 else None
        self.torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self.torso_id < 0:
            self.torso_id = next((b for b in range(m.nbody) if "torso_link" in
                                  (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")), -1)

        # Effort limits by ACTUATOR ORDER (rename-robust -- robosuite-conformant joint
        # names like l_leg_hip_pitch break name matching). config list is 43-DOF
        # actuator order: legs+waist [0:15], L-arm [15:22], L-hand [22:29], R-arm
        # [29:36], R-hand [36:43]. Body actuators (ctrl_idx) are in MOTOR_ORDER (legs,
        # waist, L-arm, R-arm) -> body slice = [0:22] + [29:36] (skip L-hand).
        eff_list = config["motor_effort_limit_list"]
        self.effort_limit = np.array(list(eff_list[0:22]) + list(eff_list[29:36]))
        self._mj_model = m

        # --- Dex3 hand maps (7/side) in the MODEL's njnt order, exactly like
        # base_sim's bridge (left_hand_index/right_hand_index). The rt/dex3 state
        # we publish and the cmd we apply must be in this same order, NOT a
        # hardcoded finger order (the model lists hands thumb-first). ---
        jid_to_act = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        lqa, lva, lci, lnm, rqa, rva, rci, rnm = [], [], [], [], [], [], [], []
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if j not in jid_to_act:
                continue
            if "left_hand" in name:
                lqa.append(int(m.jnt_qposadr[j])); lva.append(int(m.jnt_dofadr[j])); lci.append(jid_to_act[j]); lnm.append(name)
            elif "right_hand" in name:
                rqa.append(int(m.jnt_qposadr[j])); rva.append(int(m.jnt_dofadr[j])); rci.append(jid_to_act[j]); rnm.append(name)
        self._lh = (np.array(lqa), np.array(lva), np.array(lci)) if len(lci) == 7 else None
        self._rh = (np.array(rqa), np.array(rva), np.array(rci)) if len(rci) == 7 else None
        self._lh_names = lnm if len(lci) == 7 else []   # hand joint names, aligned with
        self._rh_names = rnm if len(rci) == 7 else []   # the hand cmd / _lh,_rh maps
        self.has_hands = self._lh is not None and self._rh is not None
        # hand effort limits by actuator order (L-hand [22:29], R-hand [36:43]); the
        # hand maps are built in njnt order == config hand order (thumb-first).
        self._lh_eff = (np.array(eff_list[22:29]) if self._lh is not None else np.full(7, 5.0))
        self._rh_eff = (np.array(eff_list[36:43]) if self._rh is not None else np.full(7, 5.0))

    # ------------------------------------------------------------------
    def build_obs(self) -> dict:
        """Replicates base_sim.prepare_obs for the 29-DOF body (no hands)."""
        d = self.sim.data
        m = self._mj_model
        md = d._data if hasattr(d, "_data") else d
        obs = {}
        if self.free_qadr is not None:
            obs["floating_base_pose"] = np.array(md.qpos[self.free_qadr:self.free_qadr + 7])
            obs["floating_base_vel"] = np.array(md.qvel[self.free_vadr:self.free_vadr + 6])
            obs["floating_base_acc"] = np.array(md.qacc[self.free_vadr:self.free_vadr + 6])
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = np.array(md.xquat[self.torso_id])
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(m, md, mujoco.mjtObj.mjOBJ_BODY, self.torso_id, vel6, 1)
        # mj_objectVelocity returns [ang, lin]; swap to [lin, ang] (== base_sim)
        vel6[0:3], vel6[3:6] = vel6[3:6].copy(), vel6[0:3].copy()
        obs["secondary_imu_vel"] = vel6

        obs["body_q"] = np.array(md.qpos[self.qpos_adr])
        obs["body_dq"] = np.array(md.qvel[self.qvel_adr])
        obs["body_ddq"] = np.array(md.qacc[self.qvel_adr])
        obs["body_tau_est"] = np.array(md.actuator_force[self.ctrl_idx])
        if self.has_hands:
            lq, lv, _ = self._lh
            rq, rv, _ = self._rh
            obs["left_hand_q"] = np.array(md.qpos[lq])
            obs["left_hand_dq"] = np.array(md.qvel[lv])
            obs["right_hand_q"] = np.array(md.qpos[rq])
            obs["right_hand_dq"] = np.array(md.qvel[rv])
        obs["time"] = float(md.time)
        return obs

    # ------------------------------------------------------------------
    def exchange(self):
        """Build obs, publish state to the command source, and read back the per-motor
        command WITHOUT computing or writing any torque. Returns
        ``(obs, body_cmd, hand_cmds)`` where body_cmd is a MotorCommand (or None if the
        source has nothing yet) and hand_cmds is ``(lcmd, rcmd)`` or None.

        The PD law is evaluated downstream: SonicWholeBodyController routes
        (q*, dq*, kp, kd, tau_ff) to robosuite's per-part JointPosition(Velocity)
        controllers, which apply it against the live joint state."""
        obs = self.build_obs()
        self.last_obs = obs
        self.src.update(obs)
        cmd = self.src.read()
        hands = None
        if self.has_hands and hasattr(self.src, "read_hands"):
            hands = self.src.read_hands()
        return obs, cmd, hands

