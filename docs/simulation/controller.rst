Controller
==========

Every ``Robot`` is equipped with a controller, which determines both the action space as well as how its
values are mapped into command torques. By default, all controllers have a pre-defined set of methods and
properities, though specific controllers may extend and / or override the default functionality found in
the base class.



Composite Controllers
----------------------
Composite controllers are controllers that are composed of multiple sub-controllers. They are used to control the entire robot, including all of its parts. What happens is that when an action vector is commanded to the robot, the action will be split into multipl sub-actions, each of which will be sent to the corresponding sub-controller. To understand the action split, use the function `robosuite.robots.robot.print_action_info()`. To create the action easily, we also provide a helper function  `robosuite.robots.robot.create_action_vector()` which takes the action dictionary as inputs and return the action vector with correct dimensions. For controller actions whose input dimentions does not match the robot's degrees of freedoms, you need to write your own `create_action_vector()` function inside the custom composite controller so that the robot's function can retrieve the information properly.

Composite Controller (Base class)
*********************************
.. autoclass:: robosuite.controllers.composite.composite_controller.CompositeController

  .. automethod:: load_controller_config
  .. automethod:: _init_controllers
  .. automethod:: _validate_composite_controller_specific_config
  .. automethod:: setup_action_split_idx
  .. automethod:: set_goal
  .. automethod:: reset
  .. automethod:: run_controller
  .. automethod:: get_control_dim
  .. automethod:: get_controller_base_pose
  .. automethod:: update_state
  .. automethod:: get_controller
  .. autoproperty:: action_limits

HybridMobileBase
****************
.. autoclass:: robosuite.controllers.composite.composite_controller.HybridMobileBase

  .. automethod:: set_goal
  .. autoproperty:: action_limits
  .. automethod:: create_action_vector

WholeBody
*********
.. autoclass:: robosuite.controllers.composite.composite_controller.WholeBody

  .. automethod:: _init_controllers
  .. automethod:: _init_joint_action_policy
  .. automethod:: setup_action_split_idx
  .. automethod:: setup_whole_body_controller_action_split_idx
  .. automethod:: set_goal
  .. automethod:: update_state
  .. autoproperty:: action_limits
  .. automethod:: create_action_vector
  .. automethod:: print_action_info
  .. automethod:: print_action_info_dict

WholeBodyIK
***********
.. autoclass:: robosuite.controllers.composite.composite_controller.WholeBodyIK

  .. automethod:: _validate_composite_controller_specific_config
  .. automethod:: _init_joint_action_policy




Part Controllers
-----------------
Part controllers are equivalent to controllers in robosuite up to `v1.4`. Starting from `v1.5`, we need to accommodate the diverse embodiments, and the original controllers are changed to controllers for specific parts, such as arms, heads, legs, torso, etc. 

Controller (Base class)
************************
.. autoclass:: robosuite.controllers.parts.controller.Controller

  .. automethod:: run_controller
  .. automethod:: scale_action
  .. automethod:: update_reference_data
  .. automethod:: _update_single_reference
  .. automethod:: update
  .. automethod:: update_base_pose
  .. automethod:: update_origin
  .. automethod:: update_initial_joints
  .. automethod:: clip_torques
  .. automethod:: reset_goal
  .. automethod:: nums2array
  .. autoproperty:: torque_compensation
  .. autoproperty:: actuator_limits
  .. autoproperty:: control_limits
  .. autoproperty:: name

Joint Position Controller (generic)
************************************
.. autoclass:: robosuite.controllers.parts.generic.joint_pos.JointPositionController

  .. automethod:: update_base_pose
  .. automethod:: set_goal
  .. automethod:: run_controller
  .. automethod:: reset_goal
  .. autoproperty:: control_limits
  .. autoproperty:: name

Joint Velocity Controller (generic)
************************************
.. autoclass:: robosuite.controllers.parts.generic.joint_vel.JointVelocityController

  .. automethod:: set_goal
  .. automethod:: run_controller
  .. automethod:: reset_goal
  .. autoproperty:: name


Joint Torque Controller (generic)
**********************************
.. autoclass:: robosuite.controllers.parts.generic.joint_tor.JointTorqueController

  .. automethod:: set_goal
  .. automethod:: run_controller
  .. automethod:: reset_goal
  .. autoproperty:: name


Operational Space Controller (arm)
**********************************
.. autoclass:: osc.OperationalSpaceController

  .. automethod:: set_goal
  .. automethod:: world_to_origin_frame
  .. automethod:: compute_goal_pos
  .. automethod:: goal_origin_to_eef_pose
  .. automethod:: compute_goal_orientation
  .. automethod:: run_controller
  .. automethod:: update_origin
  .. automethod:: update_initial_joints
  .. automethod:: reset_goal
  .. autoproperty:: control_limits
  .. autoproperty:: nam


Inverse Kinematics Controller (arm)
************************************
.. autoclass:: robosuite.controllers.parts.arm.ik.InverseKinematicsController

  .. automethod:: get_control
  .. automethod:: compute_joint_positions
  .. automethod:: set_goal
  .. automethod:: run_controller
  .. automethod:: update_initial_joints
  .. automethod:: reset_goal
  .. automethod:: _clip_ik_input
  .. automethod:: _make_input
  .. automethod:: _get_current_error
  .. autoproperty:: control_limits
  .. autoproperty:: name


Mobile Base Controller (mobile base)
*************************************
.. autoclass:: robosuite.controllers.parts.mobile_base.mobile_base_controller.MobileBaseController

  .. automethod:: get_base_pose
  .. automethod:: reset
  .. automethod:: run_controller
  .. automethod:: scale_action
  .. automethod:: update
  .. automethod:: update_initial_joints
  .. automethod:: clip_torques
  .. automethod:: reset_goal
  .. automethod:: nums2array
  .. autoproperty:: torque_compensation
  .. autoproperty:: actuator_limits
  .. autoproperty:: control_limits
  .. autoproperty:: name


Mobile Base Joint Velocity Controller (mobile base)
****************************************************
.. autoclass:: robosuite.controllers.parts.mobile_base.joint_vel.MobileBaseJointVelocityController

  .. automethod:: set_goal
  .. automethod:: run_controller
  .. automethod:: reset_goal
  .. autoproperty:: control_limits
  .. autoproperty:: name


Gripper Controller (base class)
********************************
.. autoclass:: robosuite.controllers.parts.gripper.gripper_controller.GripperController

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


Simple Grip Controller (gripper)
*********************************
.. autoclass:: robosuite.controllers.parts.gripper.simple_grip.SimpleGripController

  .. automethod:: set_goal
  .. automethod:: run_controller
  .. automethod:: reset_goal
  .. autoproperty:: control_limits
  .. autoproperty:: name
