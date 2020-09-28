Robot
=====

The Robot class defines a simulation object encapsulating a robot model, gripper model, and controller. Robosuite uses
class extensions of this base class, namely, SingleArm and Bimanual classes representing the two different types of
supported robots.

Base Robot
----------

.. autoclass:: robosuite.robots.robot.Robot

  .. automethod:: load_model
  .. automethod:: reset_sim
  .. automethod:: reset
  .. automethod:: setup_references
  .. automethod:: control
  .. automethod:: get_sensor_measurement
  .. autoproperty:: action_limits
  .. autoproperty:: torque_limits
  .. autoproperty:: action_dim
  .. autoproperty:: dof
  .. autoproperty:: joint_indexes
  .. autoproperty:: _joint_positions
  .. autoproperty:: _joint_velocities


SingleArm Robot
---------------

.. autoclass:: robosuite.robots.single_arm.SingleArm

  .. automethod:: control
  .. automethod:: grip_action
  .. autoproperty:: js_energy
  .. autoproperty:: ee_ft_integral
  .. autoproperty:: ee_force
  .. autoproperty:: ee_torque
  .. autoproperty:: _right_hand_pose
  .. autoproperty:: _right_hand_quat
  .. autoproperty:: _right_hand_total_velocity
  .. autoproperty:: _right_hand_pos
  .. autoproperty:: _right_hand_orn
  .. autoproperty:: _right_hand_vel
  .. autoproperty:: _right_hand_ang_vel


Bimanual Robot
--------------

.. autoclass:: robosuite.robots.bimanual.Bimanual

  .. automethod:: control
  .. automethod:: grip_action
  .. autoproperty:: arms
  .. autoproperty:: js_energy
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
