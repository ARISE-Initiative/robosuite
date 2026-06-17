"""SonicWholeBodyController: composite controller that drives the SonicG1 from the
external SONIC C++ stack (over DDS) instead of from robosuite actions.

Every control substep it does the DDS exchange (G1SonicController.exchange: build obs,
publish lowstate, read back the per-motor command q*/dq*/kp/kd/tau_ff), then ROUTES
that command to robosuite's per-part JointPositionControllers (arms / torso / legs /
grippers). Each part controller evaluates SONIC's PD law against the live joint state

    tau_i = tau_ff_i + kp_i*(q*_i - q_i) + kd_i*(dq*_i - dq_i)   (clipped to effort)

i.e. the control law is a first-class robosuite controller (the gravity-comp-free
JointPositionController, extended with dq*/tau_ff/effort-clip), NOT precomputed by the
SONIC bridge and rubber-stamped. This is the same dispatch shape as WholeBody/
WholeBodyMinkIK routing an IK solution through their part controllers, except the
"solver" here is the external SONIC stack streaming a per-joint PD command.

The robosuite action is ignored (control comes from DDS), so set_goal is a no-op.
Select via a composite config with "type": "SONIC_WBC"; the same SonicG1 can instead
be driven by standard controllers (OSC etc.) via a BASIC config.
"""
import os

import mujoco
import numpy as np
import yaml
from robosuite.controllers.composite.composite_controller import (
    CompositeController, register_composite_controller)
from robosuite.utils.sonic.controller import G1SonicController
from robosuite.utils.sonic.sources import DDSCommandSource


def _wbc_config_path():
    """Path to the SONIC WBC config (PD gains + effort limits) inside the installed
    gear_sonic package."""
    import gear_sonic
    return os.path.join(os.path.dirname(gear_sonic.__file__),
                        "utils", "mujoco_sim", "wbc_configs", "g1_29dof_sonic_model12.yaml")


@register_composite_controller
class SonicWholeBodyController(CompositeController):
    name = "SONIC_WBC"

    def __init__(self, sim, robot_model, grippers):
        super().__init__(sim, robot_model, grippers)
        self.config_path = _wbc_config_path()   # SONIC WBC config, located in gear_sonic
        with open(self.config_path) as f:
            self._cfg = yaml.load(f, Loader=yaml.FullLoader)
        self._src = DDSCommandSource(self._cfg)
        self._sonic = None       # built lazily on first run (sim fully compiled by then)
        self._part_plan = None   # part -> [(src_key, cmd_index) per joint, joint order]
        # Startup elastic band: a spring-damper force on the PELVIS body only (never a
        # qpos write), holding the floating base up during the ~18s C++ handoff. Released
        # manually (band_enabled -> False, e.g. a viewer key) once the policy balances.
        self._band = None
        self._pelvis_bid = None
        self.band_enabled = True
        self.last_command = None  # (cmd, hands) from the last exchange (for demo recording)

    def set_goal(self, all_action):
        pass  # control comes from DDS, not from robosuite actions

    def update_state(self):
        pass

    def reset(self):
        self._sonic = None
        self._part_plan = None
        self._band = None
        self._pelvis_bid = None
        self.band_enabled = True

    def release_band(self):
        """Drop the startup elastic band (manual handoff release). Idempotent; after this
        the policy balances the floating base on its own."""
        self.band_enabled = False

    def toggle_band(self):
        """Flip the startup band on/off (for a viewer key, mirroring base_sim's '9')."""
        self.band_enabled = not self.band_enabled

    def _prepare(self):
        """Build, once, the per-part command-routing plan and configure each part
        controller for SONIC: clip its output torque to per-motor effort and disable
        gravity compensation (the gripper sub-config can't carry these through
        robosuite's _load_arm_controllers, so we force them here)."""
        engine = self._sonic
        model = engine._mj_model
        # joint name -> (source_key, index-within-that-command, per-motor effort)
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

        # PELVIS body that carries the floating base (the engine's chosen freejoint) --
        # where the startup band applies its force. None on a fixed base (no freejoint).
        self._pelvis_bid = None
        if engine.free_qadr is not None:
            self._pelvis_bid = next(
                (int(model.jnt_bodyid[j]) for j in range(model.njnt)
                 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                 and int(model.jnt_qposadr[j]) == engine.free_qadr), None)

    def _apply_band(self, mj_data):
        """Hold the floating base up during the C++ startup/handoff with an elastic band:
        a spring-damper force + uprighting torque on the PELVIS body, written to
        mj_data.xfrc_applied[pelvis] ONLY -- never a qpos write, so task objects evolve
        under normal physics (unlike a full-state pin, which teleported everything).
        Reuses gear_sonic's ElasticBand (the same law base_sim uses), anchored at the
        robot's spawn pose. Cleared when band_enabled is False (manual release)."""
        bid = self._pelvis_bid
        if not self.band_enabled:
            mj_data.xfrc_applied[bid] = 0.0
            return
        if self._band is None:
            from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand
            self._band = ElasticBand()
            self._band.point = np.array(mj_data.xpos[bid])  # anchor at the spawn stand pose
            self._band.length = 0.0
        model = self._sonic._mj_model
        vel = np.zeros(6)  # mj_objectVelocity -> [angular(3), linear(3)] in world frame
        mujoco.mj_objectVelocity(model, mj_data, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
        # ElasticBand.Advance wants pose = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        pose = np.concatenate([mj_data.xpos[bid], mj_data.xquat[bid], vel[3:6], vel[0:3]])
        mj_data.xfrc_applied[bid] = self._band.Advance(pose)

    def run_controller(self, enabled_parts):
        if self._sonic is None:
            self._sonic = G1SonicController(self.sim, self._src, self._cfg)
        if self._part_plan is None:
            self._prepare()

        # Live-DDS startup hold: hold the floating base up with an elastic band on the
        # pelvis until the policy is balancing and the band is manually released. Object-
        # safe (force only, no qpos write). Mock/replay have no .bridge -> skipped. There
        # is NO fall recovery: a mid-episode collapse stays down (by design).
        mj_data = self.sim.data._data if hasattr(self.sim.data, "_data") else self.sim.data
        bridge = getattr(self._src, "bridge", None)
        if bridge is not None and self._pelvis_bid is not None:
            self._apply_band(mj_data)

        _obs, cmd, hands = self._sonic.exchange()
        self.last_command = (cmd, hands)
        outputs = {}
        if cmd is None:
            return outputs  # nothing from the source yet; hold current ctrl
        left_hand_cmd, right_hand_cmd = hands if hands is not None else (None, None)
        cmd_by_source = {"body": cmd, "lhand": left_hand_cmd, "rhand": right_hand_cmd}

        for part, part_ctrl in self.part_controllers.items():
            if not enabled_parts.get(part, False):
                continue
            plan = self._part_plan[part]
            if any(cmd_by_source[source_key] is None for source_key, _ in plan):
                continue  # command for this part unavailable (e.g. hands before DDS hand cmd)
            q_des = np.array([cmd_by_source[source_key].q[i] for source_key, i in plan])
            dq_des = np.array([cmd_by_source[source_key].dq[i] for source_key, i in plan])
            kp = np.array([cmd_by_source[source_key].kp[i] for source_key, i in plan])
            kd = np.array([cmd_by_source[source_key].kd[i] for source_key, i in plan])
            tau = np.array([cmd_by_source[source_key].tau[i] for source_key, i in plan])
            part_ctrl.set_pd_command(q_des, dq_des, kp, kd, tau)   # JointPositionController PD law
            outputs[part] = part_ctrl.run_controller()

        return outputs


# Convenience export for tests/scripts; "" if gear_sonic isn't installed (so `import
# robosuite` stays safe and the SONIC_WBC tests' skipif resolves correctly).
try:
    WBC_CONFIG = _wbc_config_path()
except Exception:
    WBC_CONFIG = ""
