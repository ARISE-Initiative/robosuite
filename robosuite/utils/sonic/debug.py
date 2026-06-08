"""DDSDebug: per-timestep recorder of the DDS-level state the controller exchanges.

When ``G1SonicController.debug`` is set, the controller hands every control step to
a DDSDebug instance: the obs we publish, the per-motor command we apply, the torque
actually written to the actuators, and the full sim state (qpos/qvel). Dumped to a
compact ``.npz`` it becomes a *golden* trajectory; the golden test suite replays the
recorded command stream deterministically and asserts our backend reproduces the
recorded obs / torque / state (see ``tests/``).

This is the recording half; ``sources.ReplayCommandSource`` is the playback half.
"""

import numpy as np

# obs dict keys we snapshot (kinematic ones are compared exactly in the tests;
# the *_ddq / *_acc / tau_est fields depend on ctrl history so they're recorded
# for completeness but only checked in the full rollout).
_OBS_KEYS = [
    "body_q", "body_dq", "body_ddq", "body_tau_est",
    "floating_base_pose", "floating_base_vel", "floating_base_acc",
    "secondary_imu_quat", "secondary_imu_vel",
]
_OBS_HAND_KEYS = ["left_hand_q", "left_hand_dq", "right_hand_q", "right_hand_dq"]


class DDSDebug:
    def __init__(self):
        self._cols = {}  # name -> list of per-step arrays
        self.meta = {}

    def _push(self, name, val):
        self._cols.setdefault(name, []).append(np.asarray(val, dtype=np.float64).copy())

    def record(self, qpos, qvel, obs, cmd, applied_tau, hand_cmds=None, hand_tau=None):
        """Snapshot one control step. cmd is a MotorCommand (body); hand_cmds is
        (left, right) MotorCommands or None; hand_tau is (left, right) applied
        torque arrays or None."""
        self._push("qpos", qpos)
        self._push("qvel", qvel)
        for k in _OBS_KEYS:
            self._push(k, obs[k])
        for k in _OBS_HAND_KEYS:
            if k in obs:
                self._push(k, obs[k])
        for f in ("q", "dq", "kp", "kd", "tau"):
            self._push(f"cmd_{f}", getattr(cmd, f))
        self._push("tau_body", applied_tau)
        if hand_cmds is not None:
            lcmd, rcmd = hand_cmds
            for f in ("q", "dq", "kp", "kd", "tau"):
                self._push(f"lh_cmd_{f}", getattr(lcmd, f))
                self._push(f"rh_cmd_{f}", getattr(rcmd, f))
        if hand_tau is not None:
            self._push("tau_lh", hand_tau[0])
            self._push("tau_rh", hand_tau[1])

    @property
    def n(self):
        return len(self._cols.get("qpos", []))

    def as_dict(self):
        """Stacked per-column arrays (T, dim), as the in-memory equivalent of a
        saved gold file. Used by the live test path (compare without writing)."""
        return {k: np.stack(v) for k, v in self._cols.items()}

    def save(self, path, meta=None):
        out = self.as_dict()
        m = dict(self.meta)
        if meta:
            m.update(meta)
        out["__meta__"] = np.array(__import__("json").dumps(m))
        np.savez_compressed(path, **out)
        return path

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=True)
        gold = {k: d[k] for k in d.files if k != "__meta__"}
        if "__meta__" in d.files:
            gold["meta"] = __import__("json").loads(str(d["__meta__"]))
        return gold
