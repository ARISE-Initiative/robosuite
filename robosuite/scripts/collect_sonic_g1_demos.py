"""Drive robosuite's SonicG1 live from the SONIC C++ controller over Unitree DDS.

Run this, then start the SONIC controller (gear_sonic_deploy) in another terminal. It opens a
viewer, so it needs a display (do not set MUJOCO_GL=egl):

    python -m robosuite.scripts.collect_sonic_g1_demos --environment Lift

Engage SONIC (']' in its terminal, or A+B+X+Y on the PICO controllers for VR teleop), then press
'9' in the viewer to release the startup band that holds the pelvis up. This script only drives
the robot and records nothing; SONIC demos are
recorded with robocasa's collect_sonic_demos.py, which also streams the camera images SONIC's
data exporter needs.
"""
import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

import robosuite
import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.sonic.bridge import SonicRobosuiteBridge

SONIC_CFG = os.path.join(os.path.dirname(os.path.abspath(robosuite.__file__)),
                         "controllers", "config", "robots", "default_sonic_g1.json")


def match_base_sim_physics(model, floor_friction=1.0, floor_torsion=0.005):
    """Give the model the physics options and floor contact of SONIC's own simulator (gear_sonic's
    base_sim.py): Euler integrator, pyramidal friction cone, no fluid forces, and its floor
    friction. The timestep is not touched: robosuite takes it from macros.SIMULATION_TIMESTEP."""
    model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
    model.opt.cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    model.opt.impratio = 1.0
    model.opt.density = 0.0
    model.opt.viscosity = 0.0
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_friction[geom] = [floor_friction, floor_torsion, 0.0001]
            model.geom_solref[geom] = [0.02, 1.0]


# If the loop falls further behind the wall clock than this (seconds), it drops the lost time and
# resumes in real time rather than running faster than real time to catch up; SONIC reads the
# robot state on the wall clock, so a catch-up burst looks to it like a robot moving at 2x speed.
MAX_LAG = 0.02


def collect_sonic_demos(env, render_hz, rtf_log):
    """Step the env in real time with SONIC's joint targets until the task succeeds or the viewer
    is closed. With rtf_log, print the real-time factor (sim time / wall time) about once a
    second."""
    controller = env.robots[0].composite_controller
    bridge = SonicRobosuiteBridge(controller.sonic_config)
    no_command = np.zeros(env.action_dim)  # an all-zero action makes the controller apply no torque

    def toggle_band_on_9(key):
        if key == ord("9"):
            controller.toggle_band()
            print(f"[collect] elastic band {'on' if controller.band_enabled else 'off'}", flush=True)

    window = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data, show_left_ui=False,
                                          show_right_ui=False, key_callback=toggle_band_on_9)
    window.opt.geomgroup[0] = 0  # robosuite puts collision geoms in group 0; hide them as its renderer does
    window.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    window.cam.trackbodyid = controller.motor_map.pelvis_body_id
    window.cam.distance, window.cam.azimuth, window.cam.elevation = 3.0, 120, -20
    print("[collect] waiting for SONIC: engage it (']' in its terminal, or A+B+X+Y in VR), then "
          "press '9' in the viewer to release the band.", flush=True)

    steps_per_render = max(1, round(env.control_freq / render_hz))
    step_count = success_steps = 0
    next_step_time = time.perf_counter()
    rtf_steps, rtf_window_start = 0, next_step_time
    try:
        while window.is_running() and success_steps < 10:
            sonic_action = bridge.get_sonic_action(env)
            env.step(no_command if sonic_action is None else sonic_action)
            step_count += 1
            # Like collect_human_demonstrations, stop once the task has held for 10 steps; it only
            # counts after the band is released.
            task_done = not controller.band_enabled and env._check_success()
            success_steps = success_steps + 1 if task_done else 0
            if step_count % steps_per_render == 0:
                window.sync()
            # SONIC runs on the wall clock and the DDS exchange happens outside env.step, so this
            # loop, not the env, keeps the sim in real time.
            next_step_time += env.control_timestep
            delay = next_step_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -MAX_LAG:
                next_step_time = time.perf_counter()
            if rtf_log:
                rtf_steps += 1
                wall_elapsed = time.perf_counter() - rtf_window_start
                if wall_elapsed >= 1.0:
                    sim_elapsed = rtf_steps * env.control_timestep
                    real_time_factor = sim_elapsed / wall_elapsed
                    print(f"[real-time] {rtf_steps} steps | wall {wall_elapsed * 1e3:.0f}ms | "
                          f"sim {sim_elapsed * 1e3:.0f}ms | RTF~{real_time_factor:.2f}x"
                          + ("  <-- BEHIND real-time" if real_time_factor < 0.97 else ""), flush=True)
                    rtf_steps, rtf_window_start = 0, time.perf_counter()
    finally:
        window.close()
    print("[collect] task success." if success_steps >= 10 else "[collect] viewer closed.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="Lift",
                        help="a robosuite environment that keeps the robot's Dex3 hands (not Wipe or TwoArmPegInHole)")
    parser.add_argument("--control-freq", type=int, default=200,
                        help="control and physics rate in Hz (one physics step per control step). With the PGS "
                             "solver, 200 Hz is contact-stable and keeps real time under contact; 500 Hz falls "
                             "behind (RTF ~0.9x) once contacts pile up")
    parser.add_argument("--render-hz", type=float, default=20.0, help="viewer refresh rate")
    parser.add_argument("--rtf-log", action="store_true", help="print the real-time factor about once a second")
    args = parser.parse_args()

    macros.SIMULATION_TIMESTEP = 1 / args.control_freq  # robosuite reads it when it builds the env

    env = robosuite.make(
        args.environment, robots=["SonicG1"],
        controller_configs=load_composite_controller_config(controller=SONIC_CFG),
        control_freq=args.control_freq, has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, hard_reset=False, ignore_done=True)
    env.reset()
    match_base_sim_physics(env.sim.model._model)
    # PGS instead of the default Newton solver: under the dense hand-object contacts of a grasp,
    # Newton's sparse Hessian factorization fails ("FactorizeHessian: rank-deficient sparse
    # Hessian" / QACC NaN). robocasa-g1's SONIC collector uses PGS for the same reason. Set here,
    # not in match_base_sim_physics: at 500 Hz PGS falls behind real time and degrades tracking.
    env.sim.model._model.opt.solver = int(mujoco.mjtSolver.mjSOL_PGS)
    try:
        collect_sonic_demos(env, args.render_hz, args.rtf_log)
    except KeyboardInterrupt:
        print("\n[collect] stopped.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
