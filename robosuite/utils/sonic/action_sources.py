"""Action sources for the SONIC whole-body controller.

The SONIC refactor makes the per-motor joint-position target **q\*** the env action vector:
``SonicWholeBodyController`` consumes it (``set_goal``) and applies the PD law with the (constant,
per-motor) gains the deployment uses; the **source** of that action is pluggable:

  - ``DDSActionSource``  -- live: publishes lowstate to / reads lowcmd from the C++ SONIC controller
                            over Unitree DDS, and returns its commanded q\* as the action.
  - ``ReplayActionSource`` -- headless: yields recorded q\* actions from a demo (Phase B).
  - a learned policy is itself a source (it outputs q\*); no class needed.

Why q\* alone is enough: the SONIC command's kp/kd are **constant** over an episode (verified: zero
variance in the gold streams), dq\*≡0 and tau_ff≡0. So gains are not in the action; the live source
**captures them once** from the first real command and hands them to the controller
(``set_command_gains``). Action layout (MOTOR order): ``[body q* (29), left-hand q* (7),
right-hand q* (7)]`` = 43 for SonicG1.
"""
import numpy as np

from robosuite.utils.sonic.controller import G1SonicController
from robosuite.utils.sonic.sources import DDSCommandSource


class SonicActionSource:
    """Interface: ``act(env) -> np.ndarray | None`` returns the per-motor q\* action (None until the
    source has a command, e.g. before the live C++ engages). ``reset(env)`` clears per-episode state.
    Live sources also expose ``gains`` -- the captured constant (kp, kd) per source -- once known."""

    gains = None  # dict: {"body": (kp, kd), "lhand": (kp, kd), "rhand": (kp, kd)}; entries appear as captured

    def reset(self, env):
        pass

    def act(self, env):
        raise NotImplementedError


class DDSActionSource(SonicActionSource):
    """Live action source: drives the SONIC C++ controller over DDS and returns its commanded q\*.

    Owns its own ``G1SonicController`` engine (for ``build_obs`` + the motor-index maps) and a
    ``DDSCommandSource`` (the Unitree SDK2 bridge). Each ``act`` does one exchange (publish lowstate,
    read lowcmd) and assembles the 43-dim q\* action. Returns None before the C++ sends commands, so
    the driver holds (the controller's startup elastic band keeps the pelvis up meanwhile)."""

    def __init__(self, config):
        self._cfg = config
        self._src = DDSCommandSource(config)   # opens the DDS bridge (no hold_q: None until engaged)
        self._engine = None                    # built lazily on first act (sim fully compiled by then)
        self.gains = None

    def reset(self, env):
        self._engine = None
        self.gains = None
        bridge = getattr(self._src, "bridge", None)
        if bridge is not None and hasattr(bridge, "reset"):
            bridge.reset()

    def _capture_gains(self, cmd, lh, rh):
        """Record the (constant) per-source gains the first time each is available."""
        if self.gains is None:
            self.gains = {}
        if "body" not in self.gains and cmd is not None:
            self.gains["body"] = (np.array(cmd.kp), np.array(cmd.kd))
        if "lhand" not in self.gains and lh is not None:
            self.gains["lhand"] = (np.array(lh.kp), np.array(lh.kd))
        if "rhand" not in self.gains and rh is not None:
            self.gains["rhand"] = (np.array(rh.kp), np.array(rh.kd))

    def act(self, env):
        if self._engine is None:
            self._engine = G1SonicController(env.sim, self._src, self._cfg)
        obs, cmd, hands = self._engine.exchange()
        if cmd is None:
            return None  # C++ not engaged yet
        lh, rh = hands if hands is not None else (None, None)
        self._capture_gains(cmd, lh, rh)
        # hands lag the body command on startup -> hold the current hand pose until commanded
        lh_q = lh.q if lh is not None else obs.get("left_hand_q")
        rh_q = rh.q if rh is not None else obs.get("right_hand_q")
        parts = [np.asarray(cmd.q, dtype=float)]
        if self._engine.has_hands:
            parts += [np.asarray(lh_q, dtype=float), np.asarray(rh_q, dtype=float)]
        return np.concatenate(parts)


class ReplayActionSource(SonicActionSource):
    """Headless action source: replays a recorded q\* action sequence (one per control step), with
    the constant gains captured at collection. Drives action-replay / dataset playback through the
    same controller path as live (Phase B)."""

    def __init__(self, actions, gains=None):
        self._actions = np.asarray(actions, dtype=float)
        self.gains = gains  # {"body": (kp,kd), ...} saved at collection; None -> controller must have them
        self._t = 0

    def reset(self, env):
        self._t = 0

    def act(self, env):
        if self._t >= len(self._actions):
            return None  # sequence exhausted
        a = self._actions[self._t]
        self._t += 1
        return a
