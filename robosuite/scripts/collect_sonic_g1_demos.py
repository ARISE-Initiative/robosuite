"""Collect demonstrations with the native robosuite SonicG1, driven by SONIC.

This is the NATIVE-robosuite analogue of sonic_robosuite/collect_sonic_demos.py: it
uses the registered ``SonicG1`` robot (true free-floating base) + the ``SONIC_WBC``
composite controller (per-part JointPositionController PD) inside a real robosuite
``Environment``, wrapped in robosuite's ``DataCollectionWrapper`` so every sim step is
recorded and packed into a robosuite-format ``demo.hdf5``.

The robot is dropped into a table-free ``EmptyArena`` (``SonicArenaEnv`` below) so the
locomotion/whole-body motions are not obstructed by a manipulation table.

Driving (the per-motor PD command stream) comes from a CommandSource:
  --mode dds     : LIVE interactive backend driven by the C++ SONIC controller over
                   Unitree DDS, running in ANOTHER terminal. This process publishes
                   lowstate + applies the received lowcmd, holds the stand pose until
                   the C++ engages, and recovers (snap to stand) if the robot falls.
                   Closed-loop -> stays balanced -> tracks whatever you drive ('T'/'N'/'P').
                   (Recording to hdf5 in this mode is a TODO; this runs the live loop.)
  --mode replay  : deterministic golden command stream (no C++); records a demo.hdf5.
                   Open-loop, so it drifts after a few seconds -- good for a self-test
                   of the recording pipeline, not for faithful demos.
  --mode mock    : zero-target hold (pipeline smoke test only); records a demo.hdf5.

Examples:
  # LIVE interactive: start the C++ SONIC controller in terminal 1 (gear_sonic_deploy;
  # keyboard, OR a planner via --planner-file + --input-type zmq/gamepad/ros2), then in
  # terminal 2 (needs a display; do NOT set MUJOCO_GL=egl):
  python -m robosuite.scripts.collect_sonic_g1_demos --mode dds
  # the robot follows whatever the C++/planner sends; --motion here only sets the initial
  # stand pose. Drive from the C++ side (keyboard ']' '/'T'/'N'/'P', or the planner).

  # self-test the recording pipeline headless (no C++):
  python -m robosuite.scripts.collect_sonic_g1_demos --mode replay --motion squat_001__A359 \
      --out /tmp/sonic_demos --steps 1200 --no-render
"""
import argparse
import datetime
import json
import os
import sys

import mujoco
import numpy as np

import robosuite  # noqa: F401  (registers SonicG1 / Dex3 / SONIC_WBC)
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.sonic_whole_body_controller import (
    WBC_CONFIG, set_sonic_source_factory)
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.sonic.controller import G1SonicController
from robosuite.wrappers import DataCollectionWrapper

# gold command-stream recordings (per motion) live in the sonic_robosuite harness repo.
GOLD_DIR = "/home/ajay/code/sonic_robosuite/tests/gold"
SONIC_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(robosuite.__file__))),
                         "robosuite", "controllers", "config", "robots", "default_sonic_g1.json")
# with-hand model path; its name ("with_hand") tells DDSCommandSource to enable the
# Dex3 hand DDS topics (publish hand state / subscribe rt/dex3/*/cmd).
WITH_HAND_XML = ("/home/ajay/code/GR00T-WholeBodyControl/gear_sonic/"
                 "data/robot_model/model_data/g1/g1_29dof_with_hand.xml")


class SonicArenaEnv(ManipulationEnv):
    """Minimal table-free stage: an EmptyArena holding one legged robot, no task
    objects. A clean place to drive the SonicG1 with SONIC (no table to obstruct
    whole-body / locomotion motion). reward/success are trivial (driving, not a task)."""

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    def _check_robot_configuration(self, robots):
        pass

    def _load_model(self):
        super()._load_model()  # instantiates + loads the robot model(s)
        arena = EmptyArena()
        arena.set_origin([0, 0, 0])
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[],
        )


def match_base_sim_physics(model, floor_friction=1.0, floor_torsion=0.005):
    """Set the global MuJoCo options + floor contact to base_sim's (gold was recorded
    there): 500 Hz Euler, pyramidal cone, impratio 1, no fluid, base_sim floor friction.
    (Per-joint armature/damping/frictionloss already come from model_data via the asset
    build, so they need no override.)

    floor_friction / floor_torsion default to base_sim's exact values (tangential 1.0,
    torsional 0.005). MuJoCo combines contact friction as max(foot, floor), so RAISING
    these grips the feet harder than base_sim -- useful if SONIC's slow walk slips /
    pivots in robosuite (low torsional friction lets the planted foot spin during turns).
    Keep the defaults to stay bit-faithful to base_sim."""
    model.opt.timestep = 0.002
    model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
    model.opt.cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    model.opt.impratio = 1.0
    model.opt.density = 0.0
    model.opt.viscosity = 0.0
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_friction[g] = [floor_friction, floor_torsion, 0.0001]
            model.geom_solref[g] = [0.02, 1.0]


def set_init_pose(env, gold):
    """Place the free-floating robot in the motion's frame-0 pose: base pose +
    per-motor body/hand joint angles (name-mapped via the SONIC controller's index
    maps, so it's robust to the model's qpos layout)."""
    cc = env.robots[0].composite_controller
    if cc._sonic is None:  # build the engine now so its index maps are available
        cc._sonic = G1SonicController(cc.sim, cc._src, cc._cfg)
    s = cc._sonic
    md = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data
    if s.free_qadr is not None:
        md.qpos[s.free_qadr:s.free_qadr + 7] = gold["qpos"][0][:7]
    md.qpos[s.qpos_adr] = gold["cmd_q"][0]
    if s.has_hands:
        md.qpos[s._lh[0]] = gold["lh_cmd_q"][0]
        md.qpos[s._rh[0]] = gold["rh_cmd_q"][0]
    env.sim.forward()


def sonic_action(env):
    """The per-motor SONIC target actually commanded this step (body q* then hand q*),
    recorded as the demo 'action' (meaningful trajectory; the robosuite action input
    itself is ignored by SONIC_WBC)."""
    last = env.robots[0].composite_controller.last_command
    if last is None or last[0] is None:
        return np.zeros(env.action_dim)
    cmd, hands = last
    parts = [cmd.q]
    if hands is not None:
        parts += [hands[0].q, hands[1].q]
    a = np.concatenate(parts)
    # pad/trim to action_dim so DataCollectionWrapper is happy
    out = np.zeros(env.action_dim)
    out[:min(len(a), env.action_dim)] = a[:env.action_dim]
    return out


def gather_to_hdf5(directory, out_dir, env_name, env_info):
    """Pack per-episode state_*.npz (written by DataCollectionWrapper) into a
    robosuite-format demo.hdf5 (states + actions + per-episode model_file)."""
    import h5py
    os.makedirs(out_dir, exist_ok=True)
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")
    n = 0
    for ep in sorted(os.listdir(directory)):
        ep_dir = os.path.join(directory, ep)
        if not os.path.isdir(ep_dir):
            continue
        states, actions = [], []
        for sf in sorted(s for s in os.listdir(ep_dir) if s.startswith("state_")):
            dic = np.load(os.path.join(ep_dir, sf), allow_pickle=True)
            states.extend(dic["states"])
            actions.extend(ai["actions"] for ai in dic["action_infos"])
        if len(states) == 0:
            continue
        del states[-1]  # last state has no following action
        assert len(states) == len(actions), (len(states), len(actions))
        n += 1
        g = grp.create_group(f"demo_{n}")
        with open(os.path.join(ep_dir, "model.xml")) as mf:
            g.attrs["model_file"] = mf.read()
        g.create_dataset("states", data=np.array(states))
        g.create_dataset("actions", data=np.array(actions))
    now = datetime.datetime.now()
    grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
    grp.attrs["repository_version"] = robosuite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = json.dumps(env_info)
    f.close()
    print(f"[demos] wrote {n} episode(s) -> {hdf5_path}", flush=True)
    return hdf5_path


def render_qpos_video(qpos_seq, out_mp4, fps=30, track=True):
    """Render a recorded full-qpos sequence (from this env's layout) to an mp4, offscreen
    (needs MUJOCO_GL=egl). track=True follows the pelvis; track=False uses a FIXED world
    camera so locomotion/translation (and foot slip) are visible. For inspecting a run
    without the on-screen viewer."""
    import subprocess
    from robosuite.utils.sonic.sources import ReferenceMockSource
    qpos_seq = np.asarray(qpos_seq)
    set_sonic_source_factory(lambda c: ReferenceMockSource(
        np.zeros((1, 29)), np.array(c["MOTOR_KP"][:29], float), np.array(c["MOTOR_KD"][:29], float)))
    try:
        env = SonicArenaEnv(robots=["SonicG1"], controller_configs=load_composite_controller_config(controller=SONIC_CFG),
                            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                            control_freq=500, hard_reset=False, ignore_done=True, horizon=10_000_000)
        env.reset(); match_base_sim_physics(env.sim.model._model)
        m = env.sim.model._model; d = env.sim.data._data
        assert m.nq == qpos_seq.shape[1], f"model nq {m.nq} != qpos dim {qpos_seq.shape[1]}"
        if not track:  # hide arena walls so a fixed camera can see the robot translate
            for g in range(m.ngeom):
                if "wall" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""):
                    m.geom_rgba[g, 3] = 0.0
        r = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        if track:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = next(b for b in range(m.nbody)
                                   if "pelvis" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""))
            cam.distance, cam.azimuth, cam.elevation = 3.0, 120, -15
        else:  # FIXED world camera -- translation across the floor is visible
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [2.0, 0.0, 0.4]
            cam.distance, cam.azimuth, cam.elevation = 9.0, 120, -12
        step = max(1, int(round((1.0 / fps) / float(m.opt.timestep))))
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
             "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", out_mp4],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for t in range(0, len(qpos_seq), step):
            d.qpos[:] = qpos_seq[t]; mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam); ff.stdin.write(r.render().tobytes())
        ff.stdin.close(); ff.wait(); env.close()
    finally:
        set_sonic_source_factory(None)
    print(f"[video] wrote {out_mp4}", flush=True)
    return out_mp4


def make_source_factory(mode, motion):
    """Return (factory, gold) where factory(config)->CommandSource for the chosen mode."""
    from robosuite.utils.sonic.sources import ReferenceMockSource, ReplayCommandSource
    gold = dict(np.load(os.path.join(GOLD_DIR, motion + ".npz"), allow_pickle=True))
    if mode == "replay":
        return (lambda c: ReplayCommandSource(gold)), gold
    if mode == "mock":
        return (lambda c: ReferenceMockSource(
            np.zeros((1, 29)), np.array(c["MOTOR_KP"][:29], float),
            np.array(c["MOTOR_KD"][:29], float))), gold
    if mode == "dds":
        from robosuite.utils.sonic.sources import DDSCommandSource
        # NO hold command: the controller pins the stand pose during startup and gates
        # release on real PD gains (kp >> 0). A hold-PD command is redundant (the pin
        # holds the robot) and would confound that gate (its kp is large), so read()
        # returns None until the C++ actually speaks.
        return (lambda c: DDSCommandSource(c)), gold
    raise ValueError(mode)


def run_dds_interactive(env, gold, render=True, max_steps=None, record_qpos=False):
    """Live interactive backend: the robot is driven by the C++ SONIC controller running
    in ANOTHER terminal (this process publishes lowstate + applies the received lowcmd).

    The startup hold is handled INSIDE SonicWholeBodyController (it pins the env-placed
    stand pose until the C++ sends its first command), so this loop is just: place the
    robot at a stand, step in real time, render. There is NO fall recovery -- if the
    robot falls during teleop it stays down (by design). The robot follows whatever you
    drive from the C++ terminal (']' start; '[ENTER]'->planner, 'W'/'T'/'N'/'P').
    Ctrl-C to stop.

    RENDERING uses a passive mujoco viewer synced at ~50 Hz -- NOT robosuite's per-step
    env.render() (the OpenCVViewer's synchronous offscreen render + cv2.imshow +
    cv2.waitKey). That throttle is load-bearing: the C++ SONIC stack is paced by fixed
    WALL-CLOCK timer threads (command-writer 500 Hz, control 50 Hz, planner 10 Hz;
    g1_deploy_onnx_ref.cpp CreateRecurrentThreadEx + `time_ += control_dt_`) and its
    planner integrates the velocity command in wall-clock, so it assumes the sim advances
    in real time (base_sim and the standalone both sleep() to hold 500 Hz). A synchronous
    per-step on-screen render can't sustain 500 Hz and drags the sim below real time ->
    the planner's gait advances faster than the sim can physically execute -> the robot
    shuffles in place and barely translates (the reported bug). Syncing the viewer at
    ~50 Hz (exactly like the standalone sonic_robosuite/collect_sonic_demos.py) keeps the
    sim+DDS exchange at real-time 500 Hz; the motion-only path is unaffected because it
    does not close a loop on wall-clock-paced base velocity.

    qpos is recorded only AFTER the C++ engages (so a video doesn't begin with the
    ~18s C++-load freeze). max_steps bounds the loop. Returns (base_z, qpos traces)."""
    import time
    cc = env.robots[0].composite_controller
    if cc._sonic is None:
        cc._sonic = G1SonicController(cc.sim, cc._src, cc._cfg)
    s = cc._sonic
    md = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data
    set_init_pose(env, gold)  # place the robot at a stand; the controller freezes it here
    bridge = getattr(cc._src, "bridge", None)
    sim_dt = float(env.sim.model._model.opt.timestep)
    action = np.zeros(env.action_dim)

    # Passive viewer synced at ~50 Hz (see docstring): rendering is decoupled from the
    # real-time 500 Hz sim/DDS loop the wall-clock-paced C++ planner depends on.
    viewer = None
    render_every = max(1, int(round(0.02 / sim_dt)))  # ~50 Hz, NOT every 500 Hz step
    if render:
        import mujoco.viewer
        m = env.sim.model._model
        try:
            viewer = mujoco.viewer.launch_passive(m, md, show_left_ui=False, show_right_ui=False)
            pelvis = next((b for b in range(m.nbody) if "pelvis" in
                           (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")), -1)
            cam = viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = pelvis
            cam.distance, cam.azimuth, cam.elevation = 3.0, 120, -20
            viewer.sync()
            print("[collect] viewer open (close window or Ctrl-C to stop).", flush=True)
        except Exception as e:
            print(f"[collect] viewer unavailable ({e}); running headless. (For a window, "
                  f"ensure a display and do NOT set MUJOCO_GL=egl.)", flush=True)
            viewer = None

    started, i, base_z, qpos_seq = False, 0, [], []
    print("[collect] waiting for C++ control -- start the SONIC controller and press ']' "
          "in its terminal ...", flush=True)
    nxt = time.perf_counter()
    try:
        while max_steps is None or i < max_steps:
            env.step(action)
            i += 1
            engaged = bridge is None or bridge.low_cmd_received
            if engaged and not started:
                started = True
                print("[collect] C++ engaged -- teleop live (drive from the C++ terminal; "
                      "Ctrl-C to stop)", flush=True)
            if started:
                base_z.append(float(md.qpos[s.free_qadr + 2]))
                if record_qpos:
                    qpos_seq.append(np.array(md.qpos))
                if i % 1000 == 0:
                    print(f"[collect] live: base_z={base_z[-1]:.3f}", flush=True)
            if viewer is not None:
                if not viewer.is_running():
                    print("[collect] viewer closed; stopping...", flush=True)
                    break
                if i % render_every == 0:
                    viewer.sync()
            nxt += sim_dt
            slp = nxt - time.perf_counter()
            if slp > 0:
                time.sleep(slp)
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
    return base_z, qpos_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "mock", "dds"], default="replay")
    ap.add_argument("--motion", default="squat_001__A359",
                    help="replay/mock: the motion to perform (drives the robot). "
                         "dds: ONLY the spawn/stand + pre-engage hold pose -- the robot "
                         "follows the live C++/planner, not this motion (any frame-0 works).")
    ap.add_argument("--out", default="/tmp/sonic_g1_demos")
    ap.add_argument("--steps", type=int, default=1200, help="sim steps to record (replay/mock)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--floor-friction", type=float, default=1.0,
                    help="floor tangential friction (base_sim=1.0; raise to grip feet harder)")
    ap.add_argument("--floor-torsion", type=float, default=0.005,
                    help="floor torsional friction (base_sim=0.005; raise to reduce foot "
                         "pivot/spin on turns -- a common cause of visible 'slipping')")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    if args.mode == "dds":
        # DDSCommandSource keys off SONIC_G1_XML to enable the Dex3 hand DDS topics
        os.environ.setdefault("SONIC_G1_XML", WITH_HAND_XML)

    factory, gold = make_source_factory(args.mode, args.motion)
    set_sonic_source_factory(factory)
    cfg = load_composite_controller_config(controller=SONIC_CFG)

    base = SonicArenaEnv(
        robots=["SonicG1"], controller_configs=cfg,
        # dds drives a passive viewer at ~50 Hz inside run_dds_interactive (keeps the sim
        # real-time for the wall-clock-paced C++ planner); don't also create robosuite's
        # per-step OpenCVViewer. replay/mock keep env.render() (deterministic, not coupled
        # to wall-clock, so render rate is harmless there).
        has_renderer=(not args.no_render) and args.mode != "dds",
        has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=500, hard_reset=False,
        ignore_done=True, horizon=10_000_000,
    )

    # --- DDS: live interactive backend (drive the C++ in another terminal) ---
    if args.mode == "dds":
        base.reset()
        match_base_sim_physics(base.sim.model._model, args.floor_friction, args.floor_torsion)
        try:
            run_dds_interactive(base, gold, render=not args.no_render)
        except KeyboardInterrupt:
            print("\n[collect] stopped.", flush=True)
        finally:
            base.close()
        return

    # --- replay / mock: drive deterministically and record a demo.hdf5 ---
    tmp_dir = os.path.join(args.out, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    env = DataCollectionWrapper(base, tmp_dir)
    env_name = type(base).__name__
    env_info = {"type": "SONIC_WBC", "robot": "SonicG1", "mode": args.mode, "motion": args.motion}
    try:
        for _ in range(args.episodes):
            env.reset()
            match_base_sim_physics(base.sim.model._model, args.floor_friction, args.floor_torsion)
            set_init_pose(base, gold)
            for _ in range(args.steps):
                env.step(sonic_action(base))
                if not args.no_render:
                    env.render()
        env.close()
    except KeyboardInterrupt:
        env.close()
    finally:
        gather_to_hdf5(tmp_dir, args.out, env_name,
                       {**env_info, "env_kwargs": {"robots": ["SonicG1"]}})


if __name__ == "__main__":
    main()
