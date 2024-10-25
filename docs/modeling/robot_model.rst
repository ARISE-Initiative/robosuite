Robot Model
===========

Robot Model
-----------
The ``RobotModel`` class serves as a direct intermediary class that reads in information from a corresponding robot XML
file and also contains relevant hard-coded information from that XML. This represents an arbitrary robot optionally equipped with a mount via the ``RobotBaseModel`` class and is the core modeling component of the higher-level ``Robot`` class used in simulation.

.. autoclass:: robosuite.models.robots.robot_model.RobotModel

  .. automethod:: set_base_xpos
  .. automethod:: set_base_ori
  .. automethod:: set_joint_attribute
  .. automethod:: add_base
  .. autoproperty:: dof
  .. autoproperty:: default_base
  .. autoproperty:: default_controller_config
  .. autoproperty:: init_qpos
  .. autoproperty:: base_xpos_offset
  .. autoproperty:: _horizontal_radius
  .. autoproperty:: _important_sites
  .. autoproperty:: _important_geoms
  .. autoproperty:: _important_sensors


Manipulator Model
-----------------
The ``ManipulatorModel`` class extends from the base ``RobotModel`` class, and represents an armed, mounted robot with an optional gripper attached to its end effector. In conjunction with the corresponding ``GripperModel`` class and ``RobotBaseModel`` class, this serves as the core modeling component of the higher-level ``Manipulator`` class used in simulation.

.. autoclass:: robosuite.models.robots.manipulators.manipulator_model.ManipulatorModel

  .. automethod:: add_gripper
  .. autoproperty:: default_gripper
  .. autoproperty:: arm_type
  .. autoproperty:: base_xpos_offset
  .. autoproperty:: eef_name
  .. autoproperty:: _important_sites


Gripper Model
-------------
The ``GripperModel`` class serves as a direct intermediary class that reads in information from a corresponding gripper XML file and also contains relevant hard-coded information from that XML. In conjunction with the ``ManipulatorModel`` class, this serves as the core modeling component of the higher-level `Manipulator` class used in simulation.

.. autoclass:: robosuite.models.grippers.gripper_model.GripperModel

  .. automethod:: format_action
  .. autoproperty:: speed
  .. autoproperty:: dof
  .. autoproperty:: init_qpos
  .. autoproperty:: _important_sites
  .. autoproperty:: _important_geoms
  .. autoproperty:: _important_sensors


Robot Base Model
-----------

The ``RobotBaseModel`` class represents the base of the robot. User can use ``add_base`` method in the ``RobotModel`` class to add a base model to the robot.

There are mainly three types of base models: ``MountModel``, ``MobileBaseModel``, and ``LegBaseModel``.

.. autoclass:: robosuite.models.bases.robot_base_model.RobotBaseModel

  .. autoproperty:: top_offset
  .. autoproperty:: horizontal_radius
  .. autoproperty:: naming_prefix
  .. autoproperty:: _important_sites
  .. autoproperty:: _important_geoms
  .. autoproperty:: _important_sensors
