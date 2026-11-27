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

   import mujoco
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
   env.sim.model._model.opt.solver = int(mujoco.mjtSolver.mjSOL_PGS)  # stable under hand-object contact
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

Use MuJoCo's PGS solver at 200 Hz. With the default Newton solver, dense hand-object contact
(grasping an object) can make the sparse Hessian factorization fail
(``FactorizeHessian: rank-deficient sparse Hessian``) and crash the simulation. PGS is set after
``match_base_sim_physics`` rather than inside it, because at 500 Hz PGS falls behind real time.

``robosuite/scripts/collect_sonic_g1_demos.py`` is this loop with a viewer. It also drops lost time
when the loop falls more than 20 ms behind the wall clock, instead of running faster than real time
to catch up: SONIC reads the robot state on the wall clock, so a catch-up burst looks to it like the
robot moving too fast.

Prerequisites
-------------

``SONIC_WBC`` needs the ``gear_sonic`` Python package from GR00T-WholeBodyControl installed in the
same Python environment as **robosuite** (Python 3.10), because **robosuite** imports it in-process.
It reads SONIC's joint and effort limits from
``gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml``. The live bridge also
needs ``unitree_sdk2py`` (bundled in that repository under ``external_dependencies``) and the
compiled controller ``gear_sonic_deploy/target/release/g1_deploy_onnx_ref``. Only SONIC's default
PD gains are supported: do not pass ``--motor-kp-scale`` / ``--motor-kd-scale`` to it.

.. code-block:: sh

   # in the robosuite environment; $GROOT_WBC is the GR00T-WholeBodyControl checkout
   pip install -e "$GROOT_WBC/gear_sonic[sim]"          # [sim] pulls in pyyaml, tyro, ...
   pip install -e "$GROOT_WBC/external_dependencies/unitree_sdk2_python"

``gear_sonic`` depends on PyTorch, but nothing on the **robosuite** side uses the GPU, so the CPU
build is enough. Build the controller by following GR00T-WholeBodyControl's
`deployment install guide <https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html>`_.

.. admonition:: MuJoCo version
   :class: note

   Use MuJoCo 3.8.1 or 3.9.0 (both tested). MuJoCo 3.17 removed ``MjData.qM``, which
   **robosuite**'s controllers use, so a fresh ``pip install`` that resolves ``mujoco>=3.3.0`` to the
   newest release breaks **robosuite** in general, not only ``SonicG1``.

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

   # Terminal 2: the SONIC controller. TensorRT's lib directory must be on the library path at
   # run time (or `source scripts/setup_env.sh` first): building the engines loads
   # libnvinfer_builder_resource.so dynamically.
   cd $GROOT_WBC/gear_sonic_deploy
   export LD_LIBRARY_PATH=$TensorRT_ROOT/lib:$LD_LIBRARY_PATH
   ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example \
     --obs-config policy/release/observation_config.yaml \
     --encoder-file policy/release/model_encoder.onnx \
     --input-type keyboard --output-type zmq --disable-crc-check

The ``policy/release/`` models are downloaded separately: run ``python download_from_hf.py`` in the
GR00T-WholeBodyControl root first. On its first run the controller converts them to TensorRT engines
(``policy/release/*.trt``, one to two minutes). Those cached engines are tied to the TensorRT version
that built them, so delete them after changing TensorRT, or the controller aborts with
``Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed``. Put TensorRT 10.13
first on the library path: CUDA 12.8's ``lib64`` ships TensorRT 10.9 and can shadow it.

Until the simulator is publishing, the controller repeatedly prints ``LowState is not available,
waiting for robot to be ready``; that is expected.

Press ``]`` in terminal 2 to engage SONIC, then ``9`` in the viewer to release the startup band.
Drive the robot from terminal 2. There is no fall recovery. This script only drives the robot and
records nothing: SONIC demos are recorded with robocasa's ``collect_sonic_demos.py``, which also
streams the camera images SONIC's data exporter needs.

.. admonition:: DDS hygiene
   :class: warning

   Run exactly one SONIC controller and one simulator at a time. Kill a stale controller with
   ``pkill -9 -x g1_deploy_onnx_``: Linux truncates process names to 15 characters, so
   ``pgrep -x g1_deploy_onnx_ref`` matches nothing. Avoid ``pkill -f`` from scripts, where it can also
   match the shell running the command.

VR teleoperation
----------------

To drive the robot from a PICO headset instead of the keyboard, set up XRoboToolkit by following
GR00T-WholeBodyControl's
`VR teleop setup <https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html>`_,
then start the controller with the ZMQ manager input and the locomotion planner, plus the PICO
streamer in a third terminal:

.. code-block:: sh

   # Terminal 2: the SONIC controller, taking commands from the PICO streamer
   ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example \
     --obs-config policy/release/observation_config.yaml \
     --encoder-file policy/release/model_encoder.onnx \
     --planner-file planner/target_vel/V2/planner_sonic.onnx \
     --input-type zmq_manager --output-type all --zmq-host localhost --disable-crc-check

   # Terminal 3: the PICO streamer (in GR00T-WholeBodyControl's teleop environment)
   python gear_sonic/scripts/pico_manager_thread_server.py --manager

Stand in the calibration pose and press A+B+X+Y once to engage SONIC (pressing it again stops it),
then ``9`` in the viewer to release the band. The locomotion planner starts in ``IDLE``, where the
left stick does nothing: press A+B to step up to slow walk first. Clicking the left stick switches
to ``VR_3PT``, where your head and hands drive the upper body; align your arms with the robot's
before clicking. A+X toggles full-body ``POSE`` mode.

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

The live test imports its tracking metric from ``$SONIC_ROBOSUITE_DIR/tests/golden_common.py``,
which is not published. That module must define ``GOLD_DIR`` (the recordings) and
``tracking_metrics(body_q, base_pose, motion)``, returning a dict with ``"mean"`` (mean joint error in
radians against the reference motion) and ``"drift"`` (base drift in metres). ``body_q`` and
``base_pose`` are sampled once per physics step at 500 Hz.
