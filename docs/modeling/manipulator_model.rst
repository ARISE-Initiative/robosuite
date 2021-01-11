Manipulator Model
=================

The `ManipulatorModel` class extends from the base `RobotModel` class, and represents an armed, mounted robot with an optional gripper attached to its end effector. In conjunction with the corresponding `GripperModel` class and `MountModel` class, this serves as the core modeling component of the higher-level `Manipulator` class used in simulation.

Base Manipulator Model
----------------------

.. autoclass:: robosuite.models.robots.manipulators.manipulator_model.ManipulatorModel

  .. automethod:: add_gripper
  .. autoproperty:: default_gripper
  .. autoproperty:: arm_type
  .. autoproperty:: base_xpos_offset
  .. autoproperty:: _important_sites
  .. autoproperty:: _eef_name