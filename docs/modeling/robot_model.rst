Robot Model
===========

Robot Model
-----------
The `RobotModel` class serves as a direct intermediary class that reads in information from a corresponding robot XML
file and also contains relevant hard-coded information from that XML. This represents an arbitrary robot optionally equipped with a mount via the `MountModel` class and is the core modeling component of the higher-level `Robot` class used in simulation.

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


Manipulator Model
-----------------
The `ManipulatorModel` class extends from the base `RobotModel` class, and represents an armed, mounted robot with an optional gripper attached to its end effector. In conjunction with the corresponding `GripperModel` class and `MountModel` class, this serves as the core modeling component of the higher-level `Manipulator` class used in simulation.

.. autoclass:: robosuite.models.robots.manipulators.manipulator_model.ManipulatorModel

  .. automethod:: add_gripper
  .. autoproperty:: default_gripper
  .. autoproperty:: arm_type
  .. autoproperty:: base_xpos_offset
  .. autoproperty:: _important_sites
  .. autoproperty:: _eef_name


Gripper Model
-------------
The `GripperModel` class serves as a direct intermediary class that reads in information from a corresponding gripper XML file and also contains relevant hard-coded information from that XML. In conjunction with the `ManipulatorModel` class, this serves as the core modeling component of the higher-level `Manipulator` class used in simulation.

.. autoclass:: robosuite.models.grippers.gripper_model.GripperModel

  .. automethod:: format_action
  .. autoproperty:: speed
  .. autoproperty:: dof
  .. autoproperty:: init_qpos
  .. autoproperty:: _important_sites
  .. autoproperty:: _important_geoms
  .. autoproperty:: _important_sensors


Mount Model
-----------
The `MountModel` class serves as a direct intermediary class that reads in information from a corresponding mount XML file and also contains relevant hard-coded information from that XML. In conjunction with the `RobotModel` class, this serves as the core modeling component of the higher-level Robot class used in simulation.

.. autoclass:: robosuite.models.mounts.mount_model.MountModel

  .. autoproperty:: top_offset
  .. autoproperty:: horizontal_radius
  .. autoproperty:: _important_sites
  .. autoproperty:: _important_geoms
  .. autoproperty:: _important_sensors
