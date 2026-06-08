"""Command sources for G1SonicController.

- DDSCommandSource: publishes lowstate to / reads lowcmd from the real C++ SONIC
  controller via Unitree SDK2 DDS (reuses gear_sonic's UnitreeSdk2Bridge).
- ReferenceMockSource: replays a reference motion as PD targets (no C++); used to
  exercise the robosuite control flow / PD / obs end-to-end without the backend.
"""

import threading

import numpy as np

from .controller import CommandSource, MotorCommand

_dds_initialized = False
_dds_lock = threading.Lock()


def init_dds_once(config):
    """Call ChannelFactoryInitialize exactly once per process."""
    global _dds_initialized
    with _dds_lock:
        if _dds_initialized:
            return
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        if config.get("INTERFACE"):
            ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
        else:
            ChannelFactoryInitialize(config["DOMAIN_ID"])
        _dds_initialized = True


class DDSCommandSource(CommandSource):
    """Bridges to the C++ SONIC controller over DDS.

    Until the C++ starts sending commands, returns a local PD "hold" command
    (q=hold_q, config kp/kd) so the robot actively stands on its own legs during
    the backend's startup latency -- giving the policy a warm, realistically
    moving robot to take over (no freeze, no band)."""

    def __init__(self, config, hold_q=None, hold_kp=None, hold_kd=None):
        import os
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge

        init_dds_once(config)
        # Enable Dex3 hand topics (publish hand state + subscribe hand cmd) only
        # with the with-hands model; otherwise body-only.
        self.n_hand = 0 if os.environ.get("SONIC_NO_HANDS") else (
            7 if "with_hand" in os.environ.get("SONIC_G1_XML", "") else 0)
        cfg = dict(config)
        cfg["NUM_HAND_MOTORS"] = self.n_hand
        self.bridge = UnitreeSdk2Bridge(cfg)
        self.n = int(config["NUM_MOTORS"])
        self.hold_q = None if hold_q is None else np.asarray(hold_q, float)
        self.hold_kp = None if hold_kp is None else np.asarray(hold_kp, float)
        self.hold_kd = None if hold_kd is None else np.asarray(hold_kd, float)

    def update(self, obs: dict):
        self.bridge.PublishLowState(obs)

    def read(self):
        b = self.bridge
        with b.low_cmd_lock:
            if not b.low_cmd_received:
                if self.hold_q is not None:
                    return MotorCommand(self.hold_q, np.zeros(self.n),
                                        self.hold_kp, self.hold_kd, np.zeros(self.n))
                return None
            mc = b.low_cmd.motor_cmd
            q = np.array([mc[i].q for i in range(self.n)])
            dq = np.array([mc[i].dq for i in range(self.n)])
            kp = np.array([mc[i].kp for i in range(self.n)])
            kd = np.array([mc[i].kd for i in range(self.n)])
            tau = np.array([mc[i].tau for i in range(self.n)])
        return MotorCommand(q, dq, kp, kd, tau)

    def read_hands(self):
        """Latest Dex3 hand commands (7/side) from rt/dex3/*/cmd, or None."""
        if self.n_hand != 7:
            return None
        b = self.bridge
        with b.left_hand_cmd_lock:
            lmc = b.left_hand_cmd.motor_cmd
            lc = MotorCommand(*[np.array([getattr(lmc[i], f) for i in range(7)])
                                for f in ("q", "dq", "kp", "kd", "tau")])
        with b.right_hand_cmd_lock:
            rmc = b.right_hand_cmd.motor_cmd
            rc = MotorCommand(*[np.array([getattr(rmc[i], f) for i in range(7)])
                                for f in ("q", "dq", "kp", "kd", "tau")])
        return lc, rc


class ReplayCommandSource(CommandSource):
    """Deterministically replays a recorded per-step command stream (body + Dex3
    hands) from a golden file -- no DDS, no C++. One frame per read(); holds the
    last frame once exhausted. ``seek(t)`` jumps to a frame (used by the per-frame
    isolated checks). This is the playback half of DDSDebug recording."""

    def __init__(self, gold):
        self.q, self.dq = gold["cmd_q"], gold["cmd_dq"]
        self.kp, self.kd, self.tau = gold["cmd_kp"], gold["cmd_kd"], gold["cmd_tau"]
        self.T = self.q.shape[0]
        self.has_hands = "lh_cmd_q" in gold
        if self.has_hands:
            self._lh = {f: gold[f"lh_cmd_{f}"] for f in ("q", "dq", "kp", "kd", "tau")}
            self._rh = {f: gold[f"rh_cmd_{f}"] for f in ("q", "dq", "kp", "kd", "tau")}
        self.t = 0
        self._cur = 0

    def seek(self, t):
        self.t = int(t)

    def update(self, obs: dict):
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


class ReferenceMockSource(CommandSource):
    """Replays per-motor joint targets (motor order) as PD targets. Advances one
    frame per read(); holds the last frame once exhausted."""

    def __init__(self, motor_targets, kp, kd):
        self.targets = np.asarray(motor_targets, dtype=np.float64)  # (T, n)
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.n = self.targets.shape[1]
        self.t = 0

    @property
    def num_frames(self):
        return self.targets.shape[0]

    @property
    def exhausted(self):
        return self.t >= self.num_frames

    def update(self, obs: dict):
        pass

    def read(self):
        idx = min(self.t, self.num_frames - 1)
        q = self.targets[idx]
        cmd = MotorCommand(
            q=q, dq=np.zeros(self.n), kp=self.kp, kd=self.kd, tau=np.zeros(self.n)
        )
        self.t += 1
        return cmd
