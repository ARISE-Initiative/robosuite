Robot Model
===========

The `RobotModel` class serves as a direct intermediary class that reads in information from a corresponding robot XML
file and also contains relevant hard-coded information from that XML. This represents an arbitrary robot optionally equipped with a mount via the `MountModel` class and is the core modeling component of the higher-level `Robot` class used in simulation.

Base Robot Model
----------------

.. autoclass:: robosuite.models.robots.robot_model.RobotModel

  .. automethod:: set_base_xpos
  .. automethod:: set_base_ori
  .. automethod:: set_joint_attribute
  .. automethod:: add_mount
  .. autoproperty:: dof
  .. autoproperty:: default_mount
  .. autoproperty:: default_controller_config
  .. autoproperty:: init_qpos
  .. autoproperty:: base_xpos_offset
  .. autoproperty:: _horizontal_radius
  .. autoproperty:: _important_sites
  .. autoproperty:: _important_geoms
  .. autoproperty:: _important_sensors