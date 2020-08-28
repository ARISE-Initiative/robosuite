Robot Model
===========

The RobotModel class serves as a direct intermediary class that reads in information from a corresponding robot XML
file and also contains relevant hard-coded information from that XML. In conjunction with the GripperModel class, this
serves as the core modeling component of the higher-level Robot class used in simulation.

Base Robot Model
----------------

.. autoclass:: robosuite.models.robots.robot_model.RobotModel

  .. automethod:: add_gripper
  .. automethod:: set_base_xpos
  .. automethod:: set_base_ori
  .. automethod:: set_joint_attribute
  .. automethod:: correct_naming
  .. autoproperty:: naming_prefix
  .. autoproperty:: joints
  .. autoproperty:: eef_name
  .. autoproperty:: robot_base
  .. autoproperty:: actuators
  .. autoproperty:: contact_geoms
  .. autoproperty:: dof
  .. autoproperty:: gripper
  .. autoproperty:: default_controller_config
  .. autoproperty:: init_qpos
  .. autoproperty:: base_xpos_offset
  .. autoproperty:: arm_type
  .. autoproperty:: _root
  .. autoproperty:: _links