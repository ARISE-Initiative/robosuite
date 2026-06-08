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
import mujoco
import numpy as np
import yaml
from robosuite.controllers.composite.composite_controller import (
    CompositeController, register_composite_controller)
from robosuite.utils.sonic.controller import G1SonicController
from robosuite.utils.sonic.sources import DDSCommandSource

# SONIC PD gains / effort limits (gear_sonic model config). External dependency of
# the live SONIC stack; override via SonicWholeBodyController.config_path if needed.
WBC_CONFIG = ("/home/ajay/code/GR00T-WholeBodyControl/gear_sonic/"
              "utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml")

# Command-source injection hook: callable(config) -> CommandSource. Default = live
# DDS (needs the C++ stack); tests set a mock (no DDS) via set_sonic_source_factory.
_SOURCE_FACTORY = None


def set_sonic_source_factory(fn):
    """Override how the controller obtains its command source (default: live DDS)."""
    global _SOURCE_FACTORY
    _SOURCE_FACTORY = fn


@register_composite_controller
class SonicWholeBodyController(CompositeController):
    name = "SONIC_WBC"
    config_path = WBC_CONFIG

    def __init__(self, sim, robot_model, grippers):
        super().__init__(sim, robot_model, grippers)
        with open(self.config_path) as f:
            self._cfg = yaml.load(f, Loader=yaml.FullLoader)
        factory = _SOURCE_FACTORY or (lambda c: DDSCommandSource(c))
        self._src = factory(self._cfg)
        self._sonic = None       # built lazily on first run (sim fully compiled by then)
        self._part_plan = None   # part -> [(src_key, cmd_index) per joint, joint order]
        self._freeze_qpos = None  # stand pose: pinned during startup + the fall-recovery target
        self.fall_z = 0.2         # base_sim-style check_fall: snap back to stand if base drops here
        self.verify = False      # test hook: stash the engine reference torque each step
        self.ref_ctrl = None
        self.last_command = None  # (cmd, hands) from the last exchange (for demo recording)

    def set_goal(self, all_action):
        pass  # control comes from DDS, not from robosuite actions

    def update_state(self):
        pass

    def reset(self):
        self._sonic = None
        self._part_plan = None
        self._freeze_qpos = None
        self.ref_ctrl = None

    def _prepare(self):
        """Build, once, the per-part command-routing plan and configure each part
        controller for SONIC: clip its output torque to per-motor effort and disable
        gravity compensation (the gripper sub-config can't carry these through
        robosuite's _load_arm_controllers, so we force them here)."""
        s = self._sonic
        m = s._mj_model
        # joint name -> (src_key, index-within-that-command, per-motor effort)
        cmd_index = {n: ("body", i, float(s.effort_limit[i]))
                     for i, n in enumerate(s.motor_joint_names)}
        if s.has_hands:
            for i, n in enumerate(s._lh_names):
                cmd_index[n] = ("lhand", i, float(s._lh_eff[i]))
            for i, n in enumerate(s._rh_names):
                cmd_index[n] = ("rhand", i, float(s._rh_eff[i]))

        self._part_plan = {}
        for part, c in self.part_controllers.items():
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j))
                     for j in c.joint_index]
            plan, eff = [], []
            for n in names:
                key, idx, e = cmd_index[n]
                plan.append((key, idx)); eff.append(e)
            self._part_plan[part] = plan
            eff = np.array(eff)
            c.torque_limits = np.array([-eff, eff])     # == SONIC's effort clip
            if hasattr(c, "use_torque_compensation"):
                c.use_torque_compensation = False       # SONIC adds no gravity comp
            c.interpolator = None                        # apply the command as-is

    def run_controller(self, enabled_parts):
        if self._sonic is None:
            self._sonic = G1SonicController(self.sim, self._src, self._cfg)
        if self._part_plan is None:
            self._prepare()

        # Live-DDS startup hold + fall recovery (mock/replay have no .bridge -> skipped):
        md = self.sim.data._data if hasattr(self.sim.data, "_data") else self.sim.data
        bridge = getattr(self._src, "bridge", None)
        if bridge is not None:
            if not bridge.low_cmd_received:
                # STARTUP FREEZE: pin the env-placed stand pose + publish it as lowstate
                # until the C++ sends its first command (a static PD can't hold a free-
                # floating humanoid up over the ~18s load). Captures the stand on entry.
                if self._freeze_qpos is None:
                    self._freeze_qpos = np.array(md.qpos)
                md.qpos[:] = self._freeze_qpos
                md.qvel[:] = 0.0
                mujoco.mj_forward(self._sonic._mj_model, md)
                self._sonic.exchange()
                self.last_command = (None, None)
                return {}
            elif (self._freeze_qpos is not None and self._sonic.free_qadr is not None
                  and float(md.qpos[self._sonic.free_qadr + 2]) < self.fall_z):
                # FALL RECOVERY (base_sim check_fall): a genuine collapse (base at floor
                # level) -> snap back to the stand. Fires only below fall_z, so it never
                # triggers on normal walk/squat / never teleports the base mid-motion; it
                # keeps the robot up through the pre-']' ready-hold window (the C++'s non-
                # balancing hold would otherwise topple the free base). TEMPORARY: a proper
                # startup handoff (e.g. an elastic band released on policy-active) is TODO.
                md.qpos[:] = self._freeze_qpos
                md.qvel[:] = 0.0
                mujoco.mj_forward(self._sonic._mj_model, md)

        _obs, cmd, hands = self._sonic.exchange()
        self.last_command = (cmd, hands)
        out = {}
        if cmd is None:
            return out  # nothing from the source yet; hold current ctrl
        lcmd, rcmd = hands if hands is not None else (None, None)
        srcs = {"body": cmd, "lhand": lcmd, "rhand": rcmd}

        for part, c in self.part_controllers.items():
            if not enabled_parts.get(part, False):
                continue
            plan = self._part_plan[part]
            if any(srcs[k] is None for k, _ in plan):
                continue  # command for this part unavailable (e.g. hands before DDS hand cmd)
            qd = np.array([srcs[k].q[i] for k, i in plan])
            dqd = np.array([srcs[k].dq[i] for k, i in plan])
            kp = np.array([srcs[k].kp[i] for k, i in plan])
            kd = np.array([srcs[k].kd[i] for k, i in plan])
            tau = np.array([srcs[k].tau[i] for k, i in plan])
            c.set_pd_command(qd, dqd, kp, kd, tau)       # JointPositionController PD law
            out[part] = c.run_controller()

        if self.verify:  # test-only: the engine's own torque on the SAME state
            self.ref_ctrl = self._sonic.compute_torques(cmd, hands)
        return out
