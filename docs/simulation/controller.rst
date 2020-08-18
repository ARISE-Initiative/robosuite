Controller
==========

Base Controller
---------------

.. autoclass:: robosuite.controllers.base_controller.Controller

  .. automethod:: __init__
  .. automethod:: run_controller
  .. automethod:: scale_action
  .. automethod:: update
  .. automethod:: update_base_pose
  .. automethod:: update_initial_joints
  .. automethod:: clip_torques
  .. automethod:: reset_goal
  .. automethod:: nums2array
  .. automethod:: torque_compensation
  .. automethod:: actuator_limits
  .. automethod:: control_limits
  .. automethod:: name