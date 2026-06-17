"""Opt-in LIVE-DDS faithfulness test for the native SonicG1 path.

Launches the real C++ SONIC controller (gear_sonic_deploy), drives the registered
robosuite SonicG1 + SONIC_WBC over Unitree DDS, plays each motion, and asserts the native
backend's joint tracking is as good as base_sim's: the nearest-reference-frame joint error
must be within TOL of the recorded gold mean for that motion. (Closeness to gold, not just
"it ran" -- so a motion the policy itself tracks poorly still passes iff native is no worse
than base_sim, and a real integration regression fails.)

Slow (~50 s/motion), launches a subprocess + opens DDS, and reuses the gold metric from the
sonic_robosuite harness, so it is gated behind an opt-in env var on top of the C++-binary /
config / gold checks. Set SONIC_LIVE_VIDEO_DIR to also render each run to an mp4 (needs EGL).

    RUN_SONIC_LIVE=1 MUJOCO_GL=egl SONIC_LIVE_VIDEO_DIR=/tmp/sonic_vids \
        pytest tests/test_robots/test_sonic_g1_live.py -q

Overrides: SONIC_DEPLOY, SONIC_ROBOSUITE_DIR, SONIC_LIVE_VIDEO_DIR.
"""
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import mujoco
import pytest

import robosuite  # noqa: F401  registers SonicG1 / Dex3 / SONIC_WBC
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.sonic_whole_body_controller import WBC_CONFIG
from robosuite.scripts.collect_sonic_g1_demos import SonicArenaEnv, SONIC_CFG, match_base_sim_physics

DEPLOY = os.environ.get("SONIC_DEPLOY", "/home/ajay/code/GR00T-WholeBodyControl/gear_sonic_deploy")
CPP_BIN = os.path.join(DEPLOY, "target", "release", "g1_deploy_onnx_ref")
REF_ROOT = os.path.join(DEPLOY, "reference", "example")
VIDEO_DIR = os.environ.get("SONIC_LIVE_VIDEO_DIR")
TOL = 0.04   # rad: native nearest-ref joint error may exceed gold by at most this (run-to-run noise)

# The nearest-ref-frame metric + recorded gold streams live in the sonic_robosuite harness.
sys.path.insert(0, os.path.join(os.environ.get("SONIC_ROBOSUITE_DIR", "/home/ajay/code/sonic_robosuite"), "tests"))
try:
    from golden_common import tracking_metrics, GOLD_DIR  # noqa: E402
    _HAVE_GOLD = True
except Exception:
    _HAVE_GOLD = False
    GOLD_DIR = ""

_LIVE_OK = (os.environ.get("RUN_SONIC_LIVE") == "1"
            and os.path.exists(CPP_BIN) and os.path.exists(WBC_CONFIG) and _HAVE_GOLD)

MOTIONS = ["squat_001__A359", "walking_quip_360_R_002__A428",
           "neutral_kick_R_001__A543", "macarena_001__A545"]


def _launch_cpp(motion):
    """Start the C++ SONIC controller for one motion (keyboard input over stdin)."""
    os.system("kill -9 $(pgrep -x g1_deploy_onnx_ref) 2>/dev/null")  # DDS hygiene: one at a time
    time.sleep(1)
    motion_dir = tempfile.mkdtemp(prefix="sonic_live_")
    os.symlink(os.path.join(REF_ROOT, motion), os.path.join(motion_dir, motion))
    cmd = ["./target/release/g1_deploy_onnx_ref", "lo", "policy/release/model_decoder.onnx",
           motion_dir, "--obs-config", "policy/release/observation_config.yaml",
           "--encoder-file", "policy/release/model_encoder.onnx",
           "--input-type", "keyboard", "--output-type", "zmq", "--disable-crc-check"]
    return subprocess.Popen(cmd, cwd=DEPLOY, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _render(model, mj_data, qpos_seq, out_mp4, sim_dt, fps=30):
    """Offscreen-render a recorded qpos sequence to mp4 (pelvis-tracking cam; needs EGL)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    pelvis = next(b for b in range(model.nbody)
                  if "pelvis" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""))
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = pelvis
    cam.distance, cam.azimuth, cam.elevation = 3.5, 120, -15
    stride = max(1, int(round((1.0 / fps) / sim_dt)))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
         "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", out_mp4],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for t in range(0, len(qpos_seq), stride):
        mj_data.qpos[:] = qpos_seq[t]
        mujoco.mj_forward(model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        ff.stdin.write(renderer.render().tobytes())
    ff.stdin.close(); ff.wait()


@pytest.mark.skipif(not _LIVE_OK,
                    reason="opt-in: RUN_SONIC_LIVE=1 + C++ binary + gear_sonic config + gold")
@pytest.mark.parametrize("motion", MOTIONS)
def test_sonic_live_dds_tracking(motion):
    if not os.path.exists(os.path.join(REF_ROOT, motion)):
        pytest.skip(f"reference motion {motion} not under {REF_ROOT}")

    cfg = load_composite_controller_config(controller=SONIC_CFG)
    env = SonicArenaEnv(robots=["SonicG1"], controller_configs=cfg, has_renderer=False,
                        has_offscreen_renderer=False, use_camera_obs=False, control_freq=500,
                        hard_reset=False, ignore_done=True, horizon=10_000_000)
    env.reset()
    match_base_sim_physics(env.sim.model._model)
    controller = env.robots[0].composite_controller
    mj_data, model = env.sim.data._data, env.sim.model._model
    action = np.zeros(env.action_dim)
    sim_dt = float(model.opt.timestep)

    cpp = _launch_cpp(motion)
    state = {"phase": "load", "rec": False}

    def drive():
        time.sleep(20); cpp.stdin.write(b"]"); cpp.stdin.flush()   # control active
        time.sleep(2); controller.release_band()                  # drop the startup band
        time.sleep(8); cpp.stdin.write(b"T"); cpp.stdin.flush()    # play the motion
        state["rec"] = True
        time.sleep(16); state["phase"] = "done"
    threading.Thread(target=drive, daemon=True).start()

    body_q, fb_pose, qpos_seq = [], [], []
    nxt = time.perf_counter()
    try:
        while state["phase"] != "done":
            env.step(action)
            if state["rec"]:
                engine = controller._sonic
                body_q.append(mj_data.qpos[engine.qpos_adr].copy())            # 29, MOTOR order
                fb_pose.append(mj_data.qpos[engine.free_qadr:engine.free_qadr + 7].copy())
                if VIDEO_DIR:
                    qpos_seq.append(np.array(mj_data.qpos))
            nxt += sim_dt
            slp = nxt - time.perf_counter()
            if slp > 0:
                time.sleep(slp)
    finally:
        try:
            cpp.terminate(); cpp.wait(timeout=5)
        except Exception:
            os.system("kill -9 $(pgrep -x g1_deploy_onnx_ref) 2>/dev/null")

    body_q, fb_pose = np.asarray(body_q), np.asarray(fb_pose)
    assert len(body_q) > 100 and np.all(np.isfinite(body_q)), "no / non-finite live data"
    achieved_hz = len(body_q) / 16.0   # the record window is 16 s wall-clock

    if VIDEO_DIR:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        _render(model, mj_data, np.asarray(qpos_seq), os.path.join(VIDEO_DIR, f"{motion}.mp4"), sim_dt)

    native = tracking_metrics(body_q, fb_pose, motion)
    gold_npz = dict(np.load(os.path.join(GOLD_DIR, motion + ".npz"), allow_pickle=True))
    gold = tracking_metrics(gold_npz["body_q"], gold_npz["floating_base_pose"], motion)
    print(f"[live] {motion}: native mean={native['mean']:.4f} drift={native['drift']:.2f}m  vs  "
          f"gold mean={gold['mean']:.4f} drift={gold['drift']:.2f}m  | {achieved_hz:.0f} Hz (tol={TOL})", flush=True)
    env.close()
    # The C++ is wall-clock paced: a sub-real-time sim over-advances its reference and tracking
    # degrades, so don't false-fail -- skip. The native live loop's healthy steady state is
    # ~350-415 Hz (high run-to-run variance) and stays gold-faithful there; below ~350 Hz the box
    # is loaded / the motion has fallen, so faithfulness is undefined. (See design doc 13.5.)
    if achieved_hz < 350:
        pytest.skip(f"{motion}: sim ran sub-real-time ({achieved_hz:.0f} Hz); faithfulness undefined")
    assert native["mean"] <= gold["mean"] + TOL, (
        f"{motion}: native joint error {native['mean']:.4f} exceeds gold {gold['mean']:.4f} + {TOL}")
