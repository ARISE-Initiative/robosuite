Robot
=====

The ``Robot`` class defines a simulation object encapsulating a robot model, gripper model, and controller. Robosuite uses class extensions of this base class to model different robotic domains. The current release focuses on manipulation, and includes a ``Manipulator`` class, which itself is extended by ``SingleArm`` and ``Bimanual`` classes representing the two different types of supported robots.

Base Robot
----------
.. autoclass:: robosuite.robots.robot.Robot

  .. automethod:: _load_controller
  .. automethod:: _postprocess_part_controller_config
  .. automethod:: load_model
  .. automethod:: reset_sim
  .. automethod:: reset
  .. automethod:: setup_references
  .. automethod:: setup_observables
  .. automethod:: _create_arm_sensors
  .. automethod:: _create_base_sensors
  .. automethod:: control
  .. automethod:: check_q_limits
  .. autoproperty:: is_mobile
  .. autoproperty:: action_limits
  .. automethod:: _input2dict
  .. autoproperty:: torque_limits
  .. autoproperty:: action_dim
  .. autoproperty:: dof
  .. automethod:: pose_in_base_from_name
  .. automethod:: set_robot_joint_positions
  .. autoproperty:: js_energy
  .. autoproperty:: _joint_positions
  .. autoproperty:: _joint_velocities
  .. autoproperty:: joint_indexes
  .. autoproperty:: arm_joint_indexes
  .. automethod:: get_sensor_measurement
  .. automethod:: visualize
  .. automethod:: _visualize_grippers
  .. autoproperty:: action_limits
  .. autoproperty:: is_mobile
  .. autoproperty:: ee_ft_integral
  .. autoproperty:: ee_force
  .. autoproperty:: ee_torque
  .. autoproperty:: _hand_pose
  .. autoproperty:: _hand_quat
  .. autoproperty:: _hand_total_velocity
  .. autoproperty:: _hand_pos
  .. autoproperty:: _hand_orn
  .. autoproperty:: _hand_vel
  .. autoproperty:: _hand_ang_vel
  .. automethod:: _load_arm_controllers
  .. automethod:: enable_parts
  .. automethod:: enabled
  .. automethod:: create_action_vector
  .. automethod:: print_action_info
  .. automethod:: print_action_info_dict
  .. automethod:: get_gripper_name
  .. automethod:: has_part
  .. autoproperty:: _joint_split_idx
  .. autoproperty:: part_controllers


Fixed Base Robot
----------------
Tabletop manipulators.

.. autoclass:: robosuite.robots.fixed_base_robot.FixedBaseRobot
Mobile Base Robot
-----------------

``WheeledRobot`` and ``LeggedRobot`` are two types of mobile base robots supported in robosuite.

.. autoclass:: robosuite.robots.mobile_base_robot.MobileBaseRobot

  .. automethod:: _load_controller
  .. automethod:: load_model
  .. automethod:: reset
  .. automethod:: setup_references
  .. automethod:: control
  .. automethod:: setup_observables
  .. autoproperty:: action_limits
  .. autoproperty:: is_mobile
  .. autoproperty:: _action_split_indexes

Mobile robot
-------------
Base class for wheeled and legged robots.

.. autoclass:: robosuite.robots.mobile_robot.MobileRobot

  .. automethod:: _load_controller
  .. automethod:: _load_base_controller
  .. automethod:: _load_torso_controller
  .. automethod:: _load_head_controller
  .. automethod:: load_model
  .. automethod:: reset
  .. automethod:: setup_references
  .. automethod:: control
  .. automethod:: setup_observables
  .. automethod:: _create_base_sensors
  .. automethod:: enable_parts
  .. autoproperty:: is_mobile
  .. autoproperty:: base
  .. autoproperty:: torso
  .. autoproperty:: head
  .. autoproperty:: legs
  .. autoproperty:: _action_split_indexes

Wheeled Robot
-------------
Mobile robots with wheeled bases.

.. autoclass:: robosuite.robots.wheeled_robot.WheeledRobot

  .. automethod:: _load_controller
  .. automethod:: load_model
  .. automethod:: reset
  .. automethod:: setup_references
  .. automethod:: control
  .. automethod:: setup_observables
  .. autoproperty:: action_limits


Legged Robot
------------
Robots with legs.

.. autoclass:: robosuite.robots.legged_robot.LeggedRobot

  .. automethod:: _load_leg_controllers
  .. automethod:: _load_controller
  .. automethod:: load_model
  .. automethod:: reset
  .. automethod:: setup_references
  .. automethod:: control
  .. automethod:: setup_observables
  .. autoproperty:: action_limits
  .. autoproperty:: is_legs_actuated
  .. autoproperty:: num_leg_joints
