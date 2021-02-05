Robot
=====

The `Robot` class defines a simulation object encapsulating a robot model, gripper model, and controller. Robosuite uses class extensions of this base class to model different robotic domains. The current release focuses on manipulation, and includes a `Manipulator` class, which itself is extended by `SingleArm` and `Bimanual` classes representing the two different types of supported robots.

Base Robot
----------

.. autoclass:: robosuite.robots.robot.Robot

  .. automethod:: load_model
  .. automethod:: reset_sim
  .. automethod:: reset
  .. automethod:: setup_references
  .. automethod:: setup_observables
  .. automethod:: control
  .. automethod:: check_q_limits
  .. automethod:: visualize
  .. automethod:: pose_in_base_from_name
  .. automethod:: set_robot_joint_positions
  .. automethod:: get_sensor_measurement
  .. autoproperty:: action_limits
  .. autoproperty:: torque_limits
  .. autoproperty:: action_dim
  .. autoproperty:: dof
  .. autoproperty:: js_energy
  .. autoproperty:: joint_indexes
  .. autoproperty:: _joint_positions
  .. autoproperty:: _joint_velocities


Manipulator Robot
-----------------

.. autoclass:: robosuite.robots.manipulator.Manipulator

  .. automethod:: grip_action
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


SingleArm Robot
---------------

.. autoclass:: robosuite.robots.single_arm.SingleArm

  .. automethod:: control
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


Bimanual Robot
--------------

.. autoclass:: robosuite.robots.bimanual.Bimanual

  .. automethod:: control
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
