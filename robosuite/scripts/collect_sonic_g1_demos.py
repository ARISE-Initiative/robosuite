"""Drive the native robosuite SonicG1 live from the C++ SONIC controller over DDS.

Uses the registered ``SonicG1`` robot (true free-floating base) + the ``SONIC_WBC``
composite controller inside a real robosuite ``Environment``. The C++ SONIC stack runs
in ANOTHER terminal and drives the robot over Unitree DDS: this process publishes
``rt/lowstate`` and applies the received ``rt/lowcmd``, holding the floating base up with
an elastic band on the pelvis during the C++ handoff. Press '9' in the viewer to release
the band once the policy is balancing (and before commanding locomotion -- it anchors the
pelvis horizontally). There is NO fall recovery -- a collapse stays down (by design; this
env is mainly for interactive debugging).

By default the robot spawns in a table-free ``EmptyArena`` (``SonicArenaEnv`` below).
``--environment`` instead loads ANY registered robosuite env (Lift, Stack, TwoArmLift,
...) -- safe for object/table envs because the band never pins qpos. The env places the
robot's x,y + facing (in front of the table); only its standing joint pose is set here.

(Recording to a robosuite demo.hdf5 from the live loop is still a TODO; the recording
helpers gather_to_hdf5 / sonic_action below + DataCollectionWrapper are exercised by the
test suite, tests/test_robots/test_sonic_g1.py.)

Example -- start the C++ SONIC controller in terminal 1 (gear_sonic_deploy; keyboard, OR a
planner via --planner-file + --input-type zmq/gamepad/ros2), then in terminal 2 (needs a
display; do NOT set MUJOCO_GL=egl):
  python -m robosuite.scripts.collect_sonic_g1_demos --environment Lift
"""
import argparse
import datetime
import json
import os

import mujoco
import numpy as np

import robosuite  # noqa: F401  (registers SonicG1 / Dex3 / SONIC_WBC)
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask

SONIC_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(robosuite.__file__))),
                         "robosuite", "controllers", "config", "robots", "default_sonic_g1.json")


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

def make_env(env_name, robot, cfg, has_renderer):
    """Build the env that SONIC drives: the table-free ``SonicArenaEnv`` (default) or ANY
    registered robosuite env via ``robosuite.make`` (e.g. Lift / Stack / TwoArmLift). Real
    object/table envs are now safe -- the startup elastic band no longer pins the whole
    qpos, so task objects evolve under normal physics. The env places the robot's x,y +
    facing (in front of the table); the robot spawns standing via SonicG1.init_qpos +
    base_xpos_offset, so no separate set-pose step is needed."""
    kw = dict(robots=[robot], controller_configs=cfg, has_renderer=has_renderer,
              has_offscreen_renderer=False, use_camera_obs=False,
              hard_reset=False, ignore_done=True, horizon=10_000_000)
    if env_name in ("SonicArena", "Empty", "empty"):
        return SonicArenaEnv(**kw)
    return robosuite.make(env_name, **kw)


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


def run_dds_interactive(env, render=True, external_pace=False):
    """Live interactive driver: the robot is driven by the C++ SONIC controller running in
    ANOTHER terminal (this process publishes rt/lowstate + applies the received rt/lowcmd).
    SonicWholeBodyController owns the startup hold (an elastic band on the pelvis holds the
    floating base up during the C++ handoff -- object-safe, no qpos pin) and builds its own
    engine; the robot spawns standing via SonicG1.init_qpos. So this loop just steps in real
    time and renders. Drive from the C++ terminal (']' start; '[ENTER]'->planner, 'W'/'T'/'N'/'P').

    Press '9' in the viewer to drop the band once the policy is balancing (and before
    commanding locomotion -- the band anchors the pelvis horizontally); that also marks the
    start of the active phase. The loop then runs until the task succeeds (env._check_success
    held briefly, as in collect_human_demonstrations) or the viewer is closed / Ctrl-C.
    SonicArenaEnv has no success condition, so it runs until you stop it. NO fall recovery.

    Rendering uses a passive mujoco.viewer (cheap viewer.sync), NOT robosuite's synchronous
    per-step OpenCVViewer (env.render) -- the C++ planner integrates its velocity command in
    wall-clock and assumes the sim runs real time, so the heavy synchronous render must stay
    off the loop."""
    import time
    controller = env.robots[0].composite_controller
    mj_data = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data
    model = env.sim.model._model
    sim_dt = float(model.opt.timestep)
    action = np.zeros(env.action_dim)
    pelvis = next((b for b in range(model.nbody) if "pelvis" in
                   (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")), -1)

    viewer = None
    if render:
        # NB: `from mujoco import viewer` (not `import mujoco.viewer`) so we don't rebind the
        # module-level `mujoco` to a function-local -- the pelvis lookup above uses it.
        from mujoco import viewer as mj_viewer
        # '9' toggles the startup elastic band (mirrors base_sim): drop it once the policy is
        # balancing and before commanding locomotion (it anchors the pelvis horizontally).
        def _key_callback(key):
            if key == ord("9"):
                controller.toggle_band()
                print(f"[collect] elastic band -> {'ON' if controller.band_enabled else 'OFF'}", flush=True)
        try:
            viewer = mj_viewer.launch_passive(model, mj_data, show_left_ui=False,
                                              show_right_ui=False, key_callback=_key_callback)
            cam = viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = pelvis
            cam.distance, cam.azimuth, cam.elevation = 3.0, 120, -20
            viewer.sync()
            print("[collect] viewer open (close window or Ctrl-C to stop).", flush=True)
        except Exception as exc:
            print(f"[collect] viewer unavailable ({exc}); running headless. (For a window, "
                  f"ensure a display and do NOT set MUJOCO_GL=egl.)", flush=True)
            viewer = None

    print("[collect] waiting for C++ -- start the SONIC controller, press ']' in its terminal, "
          "then '9' in the viewer to release the band and begin.", flush=True)
    started = False
    success_hold = 0
    step_count = 0
    next_step_time = time.perf_counter()
    try:
        while True:
            env.step(action)
            step_count += 1
            if not started and not controller.band_enabled:   # band released via '9'
                started = True
                print("[collect] band released -- active (drive from the C++ terminal).", flush=True)
            if started:
                # run until the task succeeds (held briefly), like collect_human_demonstrations;
                # SonicArenaEnv._check_success is always False -> runs until viewer close / Ctrl-C.
                success_hold = success_hold + 1 if env._check_success() else 0
                if success_hold >= 10:
                    print("[collect] task success -- stopping.", flush=True)
                    break
                if step_count % 1000 == 0:
                    print(f"[collect] live: pelvis_z={float(mj_data.xpos[pelvis][2]):.3f}", flush=True)
            if viewer is not None:
                if not viewer.is_running():
                    print("[collect] viewer closed; stopping...", flush=True)
                    break
                viewer.sync()
            if not external_pace:   # else env.real_time throttles per-substep inside env.step()
                next_step_time += sim_dt
                sleep_time = next_step_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--environment", default="SonicArena",
                    help="env to drive in: 'SonicArena' (default, table-free EmptyArena) "
                         "or any registered robosuite env (e.g. Lift, Stack, TwoArmLift). "
                         "Object/table envs are safe (the band no longer pins qpos); the "
                         "env places the robot in front of its table automatically.")
    ap.add_argument("--robot", default="SonicG1", help="SonicG1 (floating) or SonicG1Fixed")
    ap.add_argument("--floor-friction", type=float, default=1.0,
                    help="floor tangential friction (base_sim=1.0; raise to grip feet harder)")
    ap.add_argument("--floor-torsion", type=float, default=0.005,
                    help="floor torsional friction (base_sim=0.005; raise to reduce foot "
                         "pivot/spin on turns -- a common cause of visible 'slipping')")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-real-time", dest="real_time", action="store_false", default=True,
                    help="disable base.py's per-substep real-time throttle (env.real_time, ON by "
                         "default) and fall back to this loop's per-step sleep -- still ~real-time, "
                         "just the legacy mechanism. (The live C++ is wall-clock paced, so keep "
                         "real-time on for live runs.)")
    ap.add_argument("--rtf-log", action="store_true",
                    help="flip console logging to DEBUG so base.py prints the [real-time] RTF line "
                         "~1x/sec (a real-time-factor sanity check).")
    args = ap.parse_args()

    if args.rtf_log:
        import robosuite.macros as macros
        macros.CONSOLE_LOGGING_LEVEL = "DEBUG"   # before make_env so initialize_time reads it

    cfg = load_composite_controller_config(controller=SONIC_CFG)

    # No robosuite per-step renderer: run_dds_interactive drives a passive viewer so the
    # sim+DDS exchange stays real-time at the wall-clock-paced C++ planner's rate.
    base = make_env(args.environment, args.robot, cfg, has_renderer=False)
    base.reset()
    match_base_sim_physics(base.sim.model._model, args.floor_friction, args.floor_torsion)
    if args.real_time:
        base.real_time = True   # base.py per-substep real-time throttle (matches the wall-clock-paced C++)
        print("[collect] real-time pacing ON by default (env.real_time per-substep throttle); "
              "pass --no-real-time for the legacy per-step sleep.", flush=True)
    try:
        run_dds_interactive(base, render=not args.no_render, external_pace=args.real_time)
    except KeyboardInterrupt:
        print("\n[collect] stopped.", flush=True)
    finally:
        base.close()


if __name__ == "__main__":
    main()
