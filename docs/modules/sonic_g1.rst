SONIC G1 Setup
===============

The **SONIC G1** is the Unitree G1 humanoid registered as a first-class **robosuite** robot
(``SonicG1`` / ``SonicG1Fixed``): a 29-DOF body (legs, 3-DOF waist, 7-DOF arms) plus two Dex3
three-finger hands (7 DOF each), for 43 actuated DOF. It can be driven either by standard
**robosuite** part controllers (e.g. ``OSC_POSE`` via a ``BASIC`` config) or by NVIDIA's external
**SONIC** GR00T whole-body controller -- an unchanged C++ policy stack (``g1_deploy_onnx_ref``)
that drives the robot over `Unitree DDS <https://github.com/unitreerobotics/unitree_sdk2>`_. The
SONIC path is wired in through the ``SONIC_WBC`` `composite controller <controllers.html>`_, which
routes the per-motor PD command streamed by the C++ stack through robosuite's per-part
``JointPositionController`` s. **robosuite** owns the physics clock; on the SONIC path the
``env.step`` action is ignored and control comes from DDS.

This page covers installing and setting up the integration. For the full design, internals, and
the open `known limitations`_, see the handoff doc referenced at the bottom of this page.

Prerequisites
-------------

The two ways of driving the robot have different requirements:

* **Standard controllers (OSC, etc.)** -- need only a working **robosuite** + MuJoCo install (see
  `Installation <../installation.html>`_). **None** of the SONIC stack is required. The
  ``SonicG1`` / ``SonicG1Fixed`` robot, the Dex3 grippers, and the assets are all part of
  **robosuite**, so a plain ``robosuite.make(...)`` with a ``BASIC`` controller works out of the box.

* **The live SONIC whole-body controller (**\ ``SONIC_WBC``\ **)** -- additionally needs the external
  `GR00T-WholeBodyControl <https://github.com/NVIDIA/GR00T-WholeBodyControl>`_ / ``gear_sonic``
  stack, locally at ``/home/ajay/code/GR00T-WholeBodyControl/``:

  - the compiled C++ deploy binary ``gear_sonic_deploy/target/release/g1_deploy_onnx_ref``
    (built per that repo's instructions; ``deploy.sh`` will build it on first run);
  - the Unitree SDK2 DDS layer (the C++ and the sim exchange over DDS on domain 0, interface
    ``lo``);
  - the SONIC model config (PD gains / effort limits)
    ``gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`` and the source model
    ``gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml``.

.. admonition:: Interpreter / virtual environment
   :class: note

   Run the robosuite side from the SONIC stack's virtual environment so the DDS / Unitree SDK
   bindings resolve: ``/home/ajay/code/GR00T-WholeBodyControl/.venv_sim/bin/python`` (managed by
   ``uv``; this is the interpreter referenced as ``$VENV`` throughout this page).

Assets
------

The G1 body and Dex3 gripper MJCF assets are generated from the SONIC ``model_data`` and are
**already committed** to the package under ``robosuite/models/assets/robots/sonic_g1/`` and
``robosuite/models/assets/grippers/sonic_dex3_{left,right}.xml`` -- so a normal install needs no
asset build step.

You only need to regenerate them if the upstream SONIC model changes (or you want to rebuild from
a different ``model_data``). The build script splits SONIC's validated 29-DOF
``g1_29dof_with_hand.xml`` into a robosuite-conformant body plus two Dex3 grippers (copying meshes,
relabeling joints/actuators to robosuite's part-classification convention) so the reassembled robot
is physically identical to the source model (43 DOF, mass preserved):

.. code-block:: sh

   # needs the GR00T model_data + meshes available locally
   python -m robosuite.scripts.build_sonic_g1_assets

Registered components
---------------------

Importing **robosuite** registers all of the following (no extra setup):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Component
     - Notes
   * - ``SonicG1``
     - 29-DOF G1 with a true free-floating 6-DOF base (default base ``NullBase``, which preserves
       the top-level pelvis ``<freejoint/>``); bimanual Dex3 grippers. Registered in
       ``robosuite/robots/__init__.py`` ``ROBOT_CLASS_MAPPING``.
   * - ``SonicG1Fixed``
     - Same robot with the floating base removed (pelvis welded, default base ``NoActuationBase``)
       -- the variant to use for fixed-stance manipulation / OSC tasks.
   * - ``SonicDex3LeftGripper`` / ``SonicDex3RightGripper``
     - Dex3 three-finger hands, 7 DOF each (``robosuite/models/grippers/sonic_dex3_gripper.py``).
   * - ``SONIC_WBC``
     - Composite controller (``name = "SONIC_WBC"``) that drives ``SonicG1`` from the external SONIC
       C++ stack over DDS (``robosuite/controllers/composite/sonic_whole_body_controller.py``).
   * - ``default_sonic_g1.json``
     - Default ``SONIC_WBC`` controller config (all parts ``JOINT_POSITION``), at
       ``robosuite/controllers/config/robots/default_sonic_g1.json``.

Quick start
-----------

Standard controllers (no SONIC stack)
*************************************

Drop ``SonicG1Fixed`` into any task env and control the arms with the ``BASIC`` (OSC) controller --
no DDS, no C++, no SONIC config:

.. code-block:: python

   import numpy as np
   import robosuite
   from robosuite.controllers import load_composite_controller_config

   cfg = load_composite_controller_config(controller="BASIC")  # OSC_POSE arms
   env = robosuite.make(
       "TwoArmLift",
       robots=["SonicG1Fixed"],
       controller_configs=cfg,
       has_renderer=False,
       has_offscreen_renderer=False,
       use_camera_obs=False,
       control_freq=20,
   )
   env.reset()
   low, _ = env.action_spec
   action = np.zeros_like(low)
   action[0] = 0.6  # +x pose delta on the right arm
   for _ in range(50):
       env.step(action)
   env.close()

SONIC whole-body controller over live DDS
******************************************

The C++ SONIC controller and the **robosuite** sim are **separate processes** that exchange over
Unitree DDS, so this is a two-terminal recipe. Start the C++ controller first, then the sim.

.. code-block:: sh

   VENV=/home/ajay/code/GR00T-WholeBodyControl/.venv_sim/bin/python

   # Terminal 1 -- the C++ SONIC controller (from gear_sonic_deploy):
   cd /home/ajay/code/GR00T-WholeBodyControl/gear_sonic_deploy
   ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example \
     --obs-config policy/release/observation_config.yaml \
     --encoder-file policy/release/model_encoder.onnx \
     --input-type keyboard --output-type zmq --disable-crc-check
   # (or `bash deploy.sh sim`, which builds if needed and defaults to the planner.)

   # Terminal 2 -- the robosuite backend (needs a display; do NOT set MUJOCO_GL=egl):
   cd /home/ajay/code/robosuite
   $VENV -m robosuite.scripts.collect_sonic_g1_demos --mode dds

Once both are up, drive the robot from Terminal 1 (e.g. press ``]`` to engage the policy, then
``T`` / ``N`` / ``P`` for recorded motions, or enter planner mode for walking).

.. admonition:: DDS hygiene and rendering
   :class: warning

   * Run exactly **one** C++ controller and **one** backend at a time. Kill a stale C++ process
     with ``kill -9 $(pgrep -x g1_deploy_onnx_ref)`` -- **never** ``pkill -f`` (it matches your own
     shell command line).
   * The on-screen viewer needs a display and must **not** set ``MUJOCO_GL=egl`` (that selects the
     headless EGL backend). Offscreen rendering does use ``MUJOCO_GL=egl``.

Deterministic self-test (no C++)
********************************

To exercise the full robot + ``SONIC_WBC`` data-collection pipeline without the SONIC stack, replay
a recorded golden command stream headlessly. This records a robosuite-format ``demo.hdf5``:

.. code-block:: sh

   python -m robosuite.scripts.collect_sonic_g1_demos \
       --mode replay --motion squat_001__A359 --no-render

Verifying the install
---------------------

Run the SonicG1 test suite. It covers registration, model assembly (43 DOF, mass preserved), OSC
via ``robosuite.make``, the ``SONIC_WBC`` PD dispatch, the floating base, and replay data
collection. With the external SONIC config + golden streams present, all **7** tests pass:

.. code-block:: sh

   VENV=/home/ajay/code/GR00T-WholeBodyControl/.venv_sim/bin/python
   cd /home/ajay/code/robosuite
   MUJOCO_GL=egl $VENV -m pytest tests/test_robots/test_sonic_g1.py -q   # 7/7

Tests that depend on the external SONIC config or golden command streams skip automatically when
those are unavailable, so the registration / assembly / OSC tests still pass on a SONIC-free
install.

Known limitations
-----------------

Motion-file playback and offline tracking on the native path reproduce the reference (base_sim)
behavior; interactive **planner** walking via the native path is a known open issue. The
install/setup of the integration is unaffected. For the full design, internals, file index, and the
status of that open issue, see ``/home/ajay/code/sonic_robosuite_design.md``.
