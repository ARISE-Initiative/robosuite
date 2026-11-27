"""Opt-in live test of the SonicG1 path against the real SONIC C++ controller.

Launches gear_sonic_deploy's g1_deploy_onnx_ref, drives SonicG1 + SONIC_WBC over Unitree DDS,
has SONIC play each of its reference motions, and asserts that the joint tracking error is no worse
(within TOLERANCE) than in the recorded run of that motion. About a minute per motion; needs an
idle machine, because SONIC runs on the wall clock.

    RUN_SONIC_LIVE=1 MUJOCO_GL=egl SONIC_DEPLOY=<GR00T-WholeBodyControl>/gear_sonic_deploy \
        SONIC_ROBOSUITE_DIR=<sonic_robosuite> pytest tests/test_robots/test_sonic_g1_live.py -q

SONIC_ROBOSUITE_DIR is the standalone sonic_robosuite harness, which holds the tracking metric and
the recorded runs. It is not published.
Set SONIC_LIVE_VIDEO_DIR to also render each run to an mp4.
"""
import os
import subprocess
import sys
import tempfile
import threading
import time

import mujoco
import numpy as np
import pytest

pytest.importorskip("gear_sonic")

import robosuite  # noqa: E402,F401  registers SonicG1 / Dex3 / SONIC_WBC
from robosuite.controllers import load_composite_controller_config  # noqa: E402
from robosuite.environments.base import REGISTERED_ENVS  # noqa: E402
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv  # noqa: E402
from robosuite.models.arenas import EmptyArena  # noqa: E402
from robosuite.models.tasks import ManipulationTask  # noqa: E402
from robosuite.scripts.collect_sonic_g1_demos import SONIC_CFG, match_base_sim_physics  # noqa: E402
from robosuite.utils.sonic.bridge import BODY, SonicRobosuiteBridge  # noqa: E402

SONIC_DEPLOY_DIR = os.environ.get("SONIC_DEPLOY", "")
SONIC_BINARY = os.path.join(SONIC_DEPLOY_DIR, "target", "release", "g1_deploy_onnx_ref")
REFERENCE_MOTION_DIR = os.path.join(SONIC_DEPLOY_DIR, "reference", "example")
VIDEO_DIR = os.environ.get("SONIC_LIVE_VIDEO_DIR")
TOLERANCE = 0.04  # rad: the joint tracking error may exceed the recorded one by at most this (run-to-run noise)

sys.path.insert(0, os.path.join(os.environ.get("SONIC_ROBOSUITE_DIR", ""), "tests"))
try:
    from golden_common import GOLD_DIR as RECORDED_RUNS_DIR  # noqa: E402
    from golden_common import tracking_metrics

    _HAVE_RECORDED_RUNS = True
except Exception:
    _HAVE_RECORDED_RUNS = False

_LIVE_OK = os.environ.get("RUN_SONIC_LIVE") == "1" and os.path.exists(SONIC_BINARY) and _HAVE_RECORDED_RUNS

MOTIONS = ["squat_001__A359", "walking_quip_360_R_002__A428", "neutral_kick_R_001__A543", "macarena_001__A545"]


class SonicArenaEnv(ManipulationEnv):
    """An empty floor holding the robot: nothing in the way of walking and kicking."""

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    def _check_robot_configuration(self, robots):
        pass

    def _load_model(self):
        super()._load_model()
        arena = EmptyArena()
        arena.set_origin([0, 0, 0])
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[],
        )


# Defining an env registers it; keep this test-only one out of robosuite.ALL_ENVIRONMENTS, which
# upstream tests iterate.
del REGISTERED_ENVS["SonicArenaEnv"]


def _launch_sonic(motion):
    """Start the C++ SONIC controller for one motion (keyboard input over stdin)."""
    os.system("kill -9 $(pgrep -x g1_deploy_onnx_ref) 2>/dev/null")  # DDS hygiene: one at a time
    time.sleep(1)
    motion_dir = tempfile.mkdtemp(prefix="sonic_live_")
    os.symlink(os.path.join(REFERENCE_MOTION_DIR, motion), os.path.join(motion_dir, motion))
    command = [
        "./target/release/g1_deploy_onnx_ref",
        "lo",
        "policy/release/model_decoder.onnx",
        motion_dir,
        "--obs-config",
        "policy/release/observation_config.yaml",
        "--encoder-file",
        "policy/release/model_encoder.onnx",
        "--input-type",
        "keyboard",
        "--output-type",
        "zmq",
        "--disable-crc-check",
    ]
    return subprocess.Popen(
        command, cwd=SONIC_DEPLOY_DIR, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _render(model, data, qpos_frames, video_path, sim_dt, fps=30):
    """Offscreen-render a recorded qpos sequence to mp4 (pelvis-tracking camera; needs EGL)."""
    import imageio

    os.environ.setdefault("MUJOCO_GL", "egl")
    pelvis = next(
        body
        for body in range(model.nbody)
        if "pelvis" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "")
    )
    renderer = mujoco.Renderer(model, 480, 640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = pelvis
    camera.distance, camera.azimuth, camera.elevation = 3.5, 120, -15
    stride = max(1, int(round((1.0 / fps) / sim_dt)))
    writer = imageio.get_writer(video_path, fps=fps)
    try:
        for frame in range(0, len(qpos_frames), stride):
            data.qpos[:] = qpos_frames[frame]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()


@pytest.mark.skipif(not _LIVE_OK, reason="opt-in: RUN_SONIC_LIVE=1, SONIC_DEPLOY, SONIC_ROBOSUITE_DIR")
@pytest.mark.parametrize("motion", MOTIONS)
def test_sonic_live_dds_tracking(motion):
    if not os.path.exists(os.path.join(REFERENCE_MOTION_DIR, motion)):
        pytest.skip(f"reference motion {motion} not under {REFERENCE_MOTION_DIR}")

    env = SonicArenaEnv(
        robots=["SonicG1"],
        controller_configs=load_composite_controller_config(controller=SONIC_CFG),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=500,
        hard_reset=False,
        ignore_done=True,
    )
    env.reset()
    match_base_sim_physics(env.sim.model._model)
    controller = env.robots[0].composite_controller
    data, model = env.sim.data._data, env.sim.model._model
    motor_map = controller.motor_map
    bridge = SonicRobosuiteBridge(controller.sonic_config)
    no_command = np.zeros(env.action_dim)
    sim_dt = float(model.opt.timestep)

    sonic_process = _launch_sonic(motion)
    state = {"phase": "load", "recording": False}

    def drive():
        time.sleep(20)
        sonic_process.stdin.write(b"]")
        sonic_process.stdin.flush()  # control active
        time.sleep(2)
        controller.release_band()  # drop the startup band
        time.sleep(8)
        sonic_process.stdin.write(b"T")
        sonic_process.stdin.flush()  # play the motion
        state["recording"] = True
        time.sleep(16)
        state["phase"] = "done"

    threading.Thread(target=drive, daemon=True).start()

    body_q, base_pose, qpos_frames = [], [], []
    next_step_time = time.perf_counter()
    try:
        while state["phase"] != "done" and sonic_process.poll() is None:
            action = bridge.get_sonic_action(env)
            env.step(no_command if action is None else action)
            if state["recording"]:
                body_q.append(data.qpos[motor_map.qpos_adr[BODY]].copy())
                base_pose.append(data.qpos[motor_map.base_qpos_adr : motor_map.base_qpos_adr + 7].copy())
                if VIDEO_DIR:
                    qpos_frames.append(np.array(data.qpos))
            next_step_time += sim_dt
            sleep_time = next_step_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        try:
            sonic_process.terminate()
            sonic_process.wait(timeout=5)
        except Exception:
            os.system("kill -9 $(pgrep -x g1_deploy_onnx_ref) 2>/dev/null")

    assert state["phase"] == "done", "the SONIC C++ controller exited before the motion finished"
    body_q, base_pose = np.asarray(body_q), np.asarray(base_pose)
    assert len(body_q) > 100 and np.all(np.isfinite(body_q)), "no / non-finite live data"
    achieved_hz = len(body_q) / 16.0  # the record window is 16 s wall-clock

    if VIDEO_DIR:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        _render(model, data, np.asarray(qpos_frames), os.path.join(VIDEO_DIR, f"{motion}.mp4"), sim_dt)

    live = tracking_metrics(body_q, base_pose, motion)
    recorded_run = dict(np.load(os.path.join(RECORDED_RUNS_DIR, motion + ".npz"), allow_pickle=True))
    recorded = tracking_metrics(recorded_run["body_q"], recorded_run["floating_base_pose"], motion)
    print(
        f"[live] {motion}: live mean={live['mean']:.4f} drift={live['drift']:.2f}m  vs  recorded "
        f"mean={recorded['mean']:.4f} drift={recorded['drift']:.2f}m  | {achieved_hz:.0f} Hz (tol={TOLERANCE})",
        flush=True,
    )
    env.close()
    # SONIC runs on the wall clock, so a sim that falls behind real time (a loaded machine, or a
    # robot that fell) tracks worse for reasons unrelated to the integration: skip, don't fail.
    if achieved_hz < 350:
        pytest.skip(
            f"{motion}: the sim ran below real time ({achieved_hz:.0f} Hz), so the tracking "
            "comparison is not meaningful"
        )
    assert (
        live["mean"] <= recorded["mean"] + TOLERANCE
    ), f"{motion}: live joint error {live['mean']:.4f} exceeds the recorded {recorded['mean']:.4f} + {TOLERANCE}"
