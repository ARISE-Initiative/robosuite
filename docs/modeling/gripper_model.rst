Gripper Model
=============

The `GripperModel` class serves as a direct intermediary class that reads in information from a corresponding gripper XML file and also contains relevant hard-coded information from that XML. In conjunction with the `ManipulatorModel` class, this serves as the core modeling component of the higher-level `Manipulator` class used in simulation.

Base Gripper Model
------------------

.. autoclass:: robosuite.models.grippers.gripper_model.GripperModel

  .. automethod:: format_action
  .. autoproperty:: speed
  .. autoproperty:: dof
  .. autoproperty:: init_qpos
  .. autoproperty:: _important_sites
  .. autoproperty:: _important_geoms
  .. autoproperty:: _important_sensors
