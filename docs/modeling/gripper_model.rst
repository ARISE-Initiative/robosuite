Gripper Model
=============

The GripperModel class serves as a direct intermediary class that reads in information from a corresponding gripper XML
file and also contains relevant hard-coded information from that XML. In conjunction with the RobotModel class, this
serves as the core modeling component of the higher-level Robot class used in simulation.

Base Gripper Model
------------------

.. autoclass:: robosuite.models.grippers.gripper_model.GripperModel

  .. automethod:: hide_visualization
  .. automethod:: format_action
  .. autoproperty:: naming_prefix
  .. autoproperty:: visualization_sites
  .. autoproperty:: sensors
  .. autoproperty:: dof
  .. autoproperty:: init_qpos
  .. autoproperty:: speed
  .. autoproperty:: _joints
  .. autoproperty:: _actuators
  .. autoproperty:: _contact_geoms
  .. autoproperty:: _visualization_geoms
  .. autoproperty:: _important_geoms
