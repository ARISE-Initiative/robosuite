Controller
==========

Every Robot is equipped with a controller, which determines both the action space as well as how its
values are mapped into command torques. By default, all controllers have a pre-defined set of methods and
properities, though specific controllers may extend and / or override the default functionality found in
the base class.

Base Controller
---------------

.. autoclass:: robosuite.controllers.base_controller.Controller

  .. automethod:: run_controller
  .. automethod:: scale_action
  .. automethod:: update
  .. automethod:: update_base_pose
  .. automethod:: update_initial_joints
  .. automethod:: clip_torques
  .. automethod:: reset_goal
  .. automethod:: nums2array
  .. autoproperty:: torque_compensation
  .. autoproperty:: actuator_limits
  .. autoproperty:: control_limits
  .. autoproperty:: name


Joint Position Controller
-------------------------

.. autoclass:: robosuite.controllers.joint_pos.JointPositionController

  .. automethod:: set_goal
  .. automethod:: reset_goal
  .. autoproperty:: control_limits



Joint Velocity Controller
-------------------------

.. autoclass:: robosuite.controllers.joint_vel.JointVelocityController

  .. automethod:: set_goal
  .. automethod:: reset_goal


Joint Torque Controller
-----------------------

.. autoclass:: robosuite.controllers.joint_tor.JointTorqueController

  .. automethod:: set_goal
  .. automethod:: reset_goal


Operation Space Controller
--------------------------

.. autoclass:: robosuite.controllers.osc.OperationalSpaceController

  .. automethod:: set_goal
  .. automethod:: reset_goal
  .. autoproperty:: control_limits


Inverse Kinematics Controller
-----------------------------

.. autoclass:: robosuite.controllers.ik.InverseKinematicsController

  .. automethod:: setup_inverse_kinematics
  .. automethod:: sync_state
  .. automethod:: sync_ik_robot
  .. automethod:: ik_robot_eef_joint_cartesian_pose
  .. automethod:: get_control
  .. automethod:: inverse_kinematics
  .. automethod:: joint_positions_for_eef_command
  .. automethod:: bullet_base_pose_to_world_pose
  .. automethod:: set_goal
  .. automethod:: reset_goal
  .. autoproperty:: control_limits
  .. automethod:: _clip_ik_input
  .. automethod:: _make_input
  .. automethod:: _get_current_error
