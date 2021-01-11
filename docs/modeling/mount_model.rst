Mount Model
===========

The `MountModel` class serves as a direct intermediary class that reads in information from a corresponding mount XML file and also contains relevant hard-coded information from that XML. In conjunction with the `RobotModel` class, this serves as the core modeling component of the higher-level Robot class used in simulation.

Base Mount Model
----------------

.. autoclass:: robosuite.models.mounts.mount_model.MountModel

  .. autoproperty:: top_offset
  .. autoproperty:: horizontal_radius
  .. autoproperty:: _important_sites
  .. autoproperty:: _important_geoms
  .. autoproperty:: _important_sensors
