"""DDSCommandSource: the live command source for G1SonicController -- publishes lowstate
to / reads lowcmd from the real C++ SONIC controller over Unitree SDK2 DDS (reuses
gear_sonic's UnitreeSdk2Bridge). The non-DDS mock/replay sources used by tests live in
the test tree (tests/test_robots/test_sonic_g1.py)."""

import threading

import numpy as np

from .controller import MotorCommand

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


class DDSCommandSource:
    """Bridges to the C++ SONIC controller over DDS.

    Until the C++ starts sending commands, returns a local PD "hold" command
    (q=hold_q, config kp/kd) so the robot actively stands on its own legs during
    the backend's startup latency -- giving the policy a warm, realistically
    moving robot to take over (no freeze, no band)."""

    def __init__(self, config, hold_q=None, hold_kp=None, hold_kd=None):
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge

        init_dds_once(config)
        # The SonicG1 always has Dex3 hands -> always publish hand state + subscribe
        # rt/dex3/{left,right}/cmd (7 motors/side).
        self.n_hand = 7
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


