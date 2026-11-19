SONIC G1 Setup
===============

``SonicG1`` is the Unitree G1 humanoid as a **robosuite** robot: a 29-DOF body (legs, 3-DOF
waist, 7-DOF arms) with a free-floating base, plus two Dex3 three-finger hands (7 DOF each), for
43 motors. It is driven by NVIDIA's **SONIC** whole-body controller, an external C++ policy stack
(``g1_deploy_onnx_ref`` in
`GR00T-WholeBodyControl <https://github.com/NVlabs/GR00T-WholeBodyControl>`_) that talks to the
simulator over `Unitree DDS <https://github.com/unitreerobotics/unitree_sdk2>`_. **robosuite**
owns the physics clock.

How it fits together
--------------------

* The ``SONIC_WBC`` `composite controller <controllers.html>`_ takes the joint-position target of
  every motor as the env action, in SONIC's motor order: 29 body motors, then 7 left-hand and 7
  right-hand motors. It applies SONIC's PD law, ``tau = kp * (target - q) - kd * dq`` clipped to
  each motor's effort limit, with SONIC's constant gains
  (``robosuite.utils.sonic.bridge.MOTOR_KP`` / ``MOTOR_KD``).
* An all-zero action means "no SONIC command" and applies no torque. A startup elastic band holds
  the pelvis at its spawn pose until ``release_band()`` is called, so the robot stays up while
  SONIC starts.
* ``robosuite.utils.sonic.bridge.SonicRobosuiteBridge`` is the live link to SONIC. Each call to
  ``get_sonic_action(env)`` publishes the robot state and returns SONIC's 43 joint targets, or
  ``None`` while SONIC is not commanding the robot. Recorded targets or a policy's output can be
  passed to ``env.step`` directly, with no DDS involved.

.. code-block:: python

   import time

   import numpy as np
   import robosuite
   import robosuite.macros as macros
   from robosuite.controllers import load_composite_controller_config
   from robosuite.scripts.collect_sonic_g1_demos import SONIC_CFG, match_base_sim_physics
   from robosuite.utils.sonic.bridge import SonicRobosuiteBridge

   macros.SIMULATION_TIMESTEP = 1 / 200  # one physics step per control step
   env = robosuite.make("Lift", robots=["SonicG1"], control_freq=200,
                        controller_configs=load_composite_controller_config(controller=SONIC_CFG),
                        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                        hard_reset=False, ignore_done=True)
   env.reset()
   match_base_sim_physics(env.sim.model._model)  # the integrator and floor contact SONIC is tuned for
   controller = env.robots[0].composite_controller
   bridge = SonicRobosuiteBridge(controller.sonic_config)
   no_command = np.zeros(env.action_dim)
   next_step_time = time.perf_counter()
   while True:
       sonic_action = bridge.get_sonic_action(env)
       env.step(no_command if sonic_action is None else sonic_action)
       next_step_time += env.control_timestep  # SONIC runs on the wall clock: keep the sim in real time
       time.sleep(max(0.0, next_step_time - time.perf_counter()))

The physics timestep must equal the control period, so that every SONIC target drives exactly one
physics step. Call ``controller.release_band()`` once SONIC is balancing the robot.
``robosuite/scripts/collect_sonic_g1_demos.py`` is this loop with a viewer.

Prerequisites
-------------

``SONIC_WBC`` needs the ``gear_sonic`` Python package from GR00T-WholeBodyControl installed: it
reads SONIC's joint and effort limits from
``gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml``. The live bridge also
needs ``unitree_sdk2py`` (bundled in that repository under ``external_dependencies``) and the
compiled controller ``gear_sonic_deploy/target/release/g1_deploy_onnx_ref``. Only SONIC's default
PD gains are supported: do not pass ``--motor-kp-scale`` / ``--motor-kd-scale`` to it.

The bridge reports SONIC as commanding only after its body command and both hand commands have
arrived. ``gear_sonic``'s ``UnitreeSdk2Bridge`` currently starts its command subscribers before it
creates their locks, so a hand command that arrives while the bridge is being built kills that
subscriber and SONIC never engages. Until that is fixed in ``gear_sonic``, start the simulator
before the SONIC controller.

Registered components
---------------------

Importing **robosuite** registers:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Component
     - Notes
   * - ``SonicG1``
     - The robot (``robosuite/models/robots/manipulators/sonic_g1_robot.py``). Its default base,
       ``NullBase``, keeps the pelvis ``<freejoint/>``.
   * - ``SonicDex3LeftGripper`` / ``SonicDex3RightGripper``
     - The Dex3 hands (``robosuite/models/grippers/sonic_dex3_gripper.py``).
   * - ``SONIC_WBC``
     - The composite controller
       (``robosuite/controllers/composite/sonic_whole_body_controller.py``).
   * - ``JOINT_POSITION_VELOCITY``
     - Part controller that takes a full joint-space PD command each step; ``SONIC_WBC`` uses it
       for every part.
   * - ``default_sonic_g1.json``
     - The ``SONIC_WBC`` controller config, at
       ``robosuite/controllers/config/robots/default_sonic_g1.json``.

Running SONIC live
------------------

The SONIC controller and the **robosuite** simulator are separate processes, so use two
terminals. ``$GROOT_WBC`` is the GR00T-WholeBodyControl checkout.

.. code-block:: sh

   # Terminal 1: the simulator (opens a viewer, so it needs a display; do not set MUJOCO_GL=egl)
   python -m robosuite.scripts.collect_sonic_g1_demos --environment Lift

   # Terminal 2: the SONIC controller
   cd $GROOT_WBC/gear_sonic_deploy
   ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example \
     --obs-config policy/release/observation_config.yaml \
     --encoder-file policy/release/model_encoder.onnx \
     --input-type keyboard --output-type zmq --disable-crc-check

Press ``]`` in terminal 2 to engage SONIC, then ``9`` in the viewer to release the startup band.
Drive the robot from terminal 2. There is no fall recovery. This script only drives the robot and
records nothing: SONIC demos are recorded with robocasa's ``collect_sonic_demos.py``, which also
streams the camera images SONIC's data exporter needs.

.. admonition:: DDS hygiene
   :class: warning

   Run exactly one SONIC controller and one simulator at a time. Kill a stale controller with
   ``kill -9 $(pgrep -x g1_deploy_onnx_ref)``, never ``pkill -f``, which also matches your own
   shell command line.

Assets
------

The robot and gripper MJCF files and meshes are committed under
``robosuite/models/assets/robots/sonic_g1/`` and ``robosuite/models/assets/grippers/``. They are
generated from SONIC's own G1 model, so the physics is identical; only joint names (to fit
robosuite's part naming) and the hand colour differ. To regenerate them after SONIC's model
changes:

.. code-block:: sh

   python -m robosuite.scripts.build_sonic_g1_assets

Verifying the install
---------------------

.. code-block:: sh

   MUJOCO_GL=egl pytest tests/test_robots/test_sonic_g1.py tests/test_robots/test_sonic_g1_default_pose.py -q

The tests cover registration, model assembly, the motor order, the PD law with its effort limits,
the startup band and the DDS bridge (against a fake DDS layer).

Two further checks compare against recorded runs of the SONIC C++ controller, which are not
shipped with robosuite, so they are skipped by default. With ``SONIC_RECORDED_RUNS_DIR`` pointing
at those recordings, ``test_sonic_g1.py`` also checks that ``SONIC_WBC`` reproduces the recorded
torques. ``tests/test_robots/test_sonic_g1_live.py`` runs the real SONIC controller and compares
its joint tracking with the recordings; it is opt-in (see its docstring) and needs an idle
machine.
