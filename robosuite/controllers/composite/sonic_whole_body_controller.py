"""SonicWholeBodyController: composite controller that applies the SONIC PD law from the **action**.

The env action is the per-motor joint-position target q* (MOTOR order: [body q*(29), left-hand q*(7),
right-hand q*(7)] = 43 for SonicG1). The controller routes it to robosuite's per-part
JointPosition(Velocity)Controllers, which evaluate the gravity-comp-free PD law against the live state

    tau_i = kp_i*(q*_i - q_i) + kd_i*(0 - dq_i)        (dq*≡0, tau_ff≡0; clipped to per-motor effort)

The action is produced by a pluggable **source** (live DDS / replay / policy) -- see
``robosuite.utils.sonic.action_sources`` -- NOT pulled from DDS inside the controller. The per-motor
kp/kd are CONSTANT over an episode (the SONIC policy modulates only q*; dq*/tau_ff are zero), so they
are not carried in the action: the live source captures them once from the first command and calls
``set_command_gains``. Until both an action and gains are available the controller holds (no torque),
while a startup elastic band on the pelvis keeps the floating base up during the C++ handoff
(force only, never a qpos write); release it via ``release_band()`` / the viewer '9' key.

``set_goal`` stores the action; ``run_controller`` routes it. Select via a composite config with
"type": "SONIC_WBC"; the same SonicG1 can instead be driven by standard controllers (OSC etc.).
"""
import os

import mujoco
import numpy as np
import yaml
from robosuite.controllers.composite.composite_controller import (
    CompositeController, register_composite_controller)
from robosuite.utils.sonic.controller import G1SonicController
from scipy.spatial.transform import Rotation


def _wbc_config_path():
    """Path to the SONIC WBC config (effort + joint limits) inside the installed gear_sonic."""
    import gear_sonic
    return os.path.join(os.path.dirname(gear_sonic.__file__),
                        "utils", "mujoco_sim", "wbc_configs", "g1_29dof_sonic_model12.yaml")


@register_composite_controller
class SonicWholeBodyController(CompositeController):
    name = "SONIC_WBC"

    def __init__(self, sim, robot_model, grippers):
        super().__init__(sim, robot_model, grippers)
        self.config_path = _wbc_config_path()
        with open(self.config_path) as f:
            self._cfg = yaml.load(f, Loader=yaml.FullLoader)
        self._maps = None        # G1SonicController used ONLY for motor maps (no DDS, no exchange)
        self._part_plan = None    # part -> [(source_key, index-within-source) per joint]
        self._action_split = None  # (n_body, n_hand) to slice the flat q* action into sources
        self._action_q = None      # latest q* action (set by set_goal)
        self._cmd_gains = {}       # {"body": (kp, kd), "lhand": (kp, kd), "rhand": (kp, kd)} (constant)
        # Startup elastic band: spring-damper force on the PELVIS body only (never a qpos write),
        # holding the floating base up during the C++ handoff. Released via release_band()/'9'.
        self._band = None
        self._band_ref_rot = None
        self._pelvis_bid = None
        self.band_enabled = True

    # --- action space: per-motor q* targets in MOTOR order [body(29), L-hand(7), R-hand(7)].
    # --- engine-free (valid before the lazy maps-engine is built): body bounds from the config
    # --- joint limits (29, body motor order); hands have no config limits, so use generous bounds
    # --- (their values are not used for faithfulness -- only the action dim is load-bearing -- and
    # --- the commanded q* is within range). ---
    def _num_hand_motors(self):
        """Per-hand actuated Dex3 motor count from the model (0 if the robot has no hands)."""
        m = self.sim.model._model if hasattr(self.sim.model, "_model") else self.sim.model
        acted = {int(m.actuator_trnid[a, 0]) for a in range(m.nu)}
        return sum(1 for j in range(m.njnt) if j in acted
                   and "left_hand" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""))

    @property
    def action_limits(self):
        c = self._cfg
        body_lo = np.array(c["motor_pos_lower_limit_list"], dtype=float)  # 29, body motor order
        body_hi = np.array(c["motor_pos_upper_limit_list"], dtype=float)
        n_hand = self._num_hand_motors()
        hand_lo = np.full(2 * n_hand, -np.pi)
        hand_hi = np.full(2 * n_hand, np.pi)
        return np.concatenate([body_lo, hand_lo]), np.concatenate([body_hi, hand_hi])

    def set_goal(self, all_action):
        self._action_q = np.asarray(all_action, dtype=float)

    def set_command_gains(self, gains):
        """Set the constant per-source PD gains, e.g. from the live source the first time it has a
        command. ``gains``: dict {"body": (kp, kd), "lhand": (kp, kd), "rhand": (kp, kd)} (any subset;
        entries accumulate). Until a part's source gains are present, that part holds."""
        if gains:
            self._cmd_gains.update(gains)

    def update_state(self):
        pass

    def reset(self):
        self._maps = None
        self._part_plan = None
        self._action_split = None
        self._action_q = None
        self._cmd_gains = {}
        self._band = None
        self._band_ref_rot = None
        self._pelvis_bid = None
        self.band_enabled = True

    def release_band(self):
        """Drop the startup elastic band (manual handoff release). Idempotent."""
        self.band_enabled = False

    def toggle_band(self):
        """Flip the startup band on/off (for a viewer key, mirroring base_sim's '9')."""
        self.band_enabled = not self.band_enabled

    def _prepare(self):
        """Build, once, the per-part action-routing plan + the flat-action split, and configure each
        part controller for SONIC (clip output torque to per-motor effort; no gravity comp)."""
        engine = self._maps
        model = engine._mj_model
        # joint name -> (source_key, index-within-that-source, per-motor effort)
        cmd_index = {name: ("body", i, float(engine.effort_limit[i]))
                     for i, name in enumerate(engine.motor_joint_names)}
        if engine.has_hands:
            for i, name in enumerate(engine._lh_names):
                cmd_index[name] = ("lhand", i, float(engine._lh_eff[i]))
            for i, name in enumerate(engine._rh_names):
                cmd_index[name] = ("rhand", i, float(engine._rh_eff[i]))

        self._part_plan = {}
        for part, part_ctrl in self.part_controllers.items():
            joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
                           for j in part_ctrl.joint_index]
            plan, efforts = [], []
            for name in joint_names:
                source_key, cmd_idx, effort = cmd_index[name]
                plan.append((source_key, cmd_idx)); efforts.append(effort)
            self._part_plan[part] = plan
            efforts = np.array(efforts)
            part_ctrl.torque_limits = np.array([-efforts, efforts])   # == SONIC's effort clip
            if hasattr(part_ctrl, "use_torque_compensation"):
                part_ctrl.use_torque_compensation = False   # SONIC adds no gravity comp
            part_ctrl.interpolator = None                    # apply the command as-is

        # how the flat action splits into per-source blocks (matches the source's concat order)
        self._action_split = (engine.num_motors, 7 if engine.has_hands else 0)

        # PELVIS body carrying the floating base (the engine's chosen freejoint) -- where the
        # startup band applies its force. None on a fixed base (no freejoint).
        self._pelvis_bid = None
        if engine.free_qadr is not None:
            self._pelvis_bid = next(
                (int(model.jnt_bodyid[j]) for j in range(model.njnt)
                 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                 and int(model.jnt_qposadr[j]) == engine.free_qadr), None)

    def _apply_band(self, mj_data):
        """Hold the floating base up during the C++ startup/handoff with an elastic band.

        The band applies a spring-damper force plus an orientation-hold torque on
        the PELVIS body, written to xfrc_applied[pelvis] ONLY. The orientation
        reference is the spawn/reset pelvis pose, so kitchens that spawn Sonic
        facing a counter are not rotated back to world-zero yaw.
        """
        bid = self._pelvis_bid
        if not self.band_enabled:
            mj_data.xfrc_applied[bid] = 0.0
            return
        if self._band is None:
            from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand
            self._band = ElasticBand()
            self._band.point = np.array(mj_data.xpos[bid])  # anchor at the spawn stand pose
            self._band.length = 0.0
            quat = mj_data.xquat[bid]
            self._band_ref_rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        model = self._maps._mj_model
        vel = np.zeros(6)  # mj_objectVelocity -> [angular(3), linear(3)] in world frame
        mujoco.mj_objectVelocity(model, mj_data, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
        lin_vel, ang_vel = vel[3:6], vel[0:3]
        quat = mj_data.xquat[bid]
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rotvec = (rot * self._band_ref_rot.inv()).as_rotvec()
        force = (
            self._band.kp_pos
            * (self._band.point - mj_data.xpos[bid] + np.array([0.0, 0.0, self._band.length]))
            - self._band.kd_pos * lin_vel
        )
        torque = -self._band.kp_ang * rotvec - self._band.kd_ang * ang_vel
        mj_data.xfrc_applied[bid] = np.concatenate([force, torque])

    def _action_by_source(self):
        """Slice the flat q* action into {body, lhand, rhand} blocks (matching the source's concat)."""
        n_body, n_hand = self._action_split
        a = self._action_q
        out = {"body": a[0:n_body]}
        if n_hand:
            out["lhand"] = a[n_body:n_body + n_hand]
            out["rhand"] = a[n_body + n_hand:n_body + 2 * n_hand]
        return out

    def run_controller(self, enabled_parts):
        if self._maps is None:
            self._maps = G1SonicController(self.sim, None, self._cfg)  # maps only; no DDS / no exchange
        if self._part_plan is None:
            self._prepare()

        # Startup hold: elastic band on the pelvis until released. Object-safe (force only).
        mj_data = self.sim.data._data if hasattr(self.sim.data, "_data") else self.sim.data
        if self._pelvis_bid is not None:
            self._apply_band(mj_data)

        outputs = {}
        if self._action_q is None or not self._cmd_gains:
            return outputs  # no command/gains yet -> hold current ctrl (band keeps the pelvis up)

        abys = self._action_by_source()
        for part, part_ctrl in self.part_controllers.items():
            if not enabled_parts.get(part, False):
                continue
            plan = self._part_plan[part]
            srcs = {sk for sk, _ in plan}
            if any(sk not in self._cmd_gains or sk not in abys for sk in srcs):
                continue  # this part's source gains/action not available yet (e.g. hands at startup)
            q_des = np.array([abys[source_key][i] for source_key, i in plan])
            kp = np.array([self._cmd_gains[source_key][0][i] for source_key, i in plan])
            kd = np.array([self._cmd_gains[source_key][1][i] for source_key, i in plan])
            dq_des = np.zeros_like(q_des)   # SONIC commands dq*≡0 (pure position PD + kd damping)
            tau = np.zeros_like(q_des)      # tau_ff≡0
            part_ctrl.set_pd_command(q_des, dq_des, kp, kd, tau)   # JointPositionController PD law
            outputs[part] = part_ctrl.run_controller()

        return outputs


# Convenience export for tests/scripts; "" if gear_sonic isn't installed (so `import robosuite`
# stays safe and the SONIC_WBC tests' skipif resolves correctly).
try:
    WBC_CONFIG = _wbc_config_path()
except Exception:
    WBC_CONFIG = ""
