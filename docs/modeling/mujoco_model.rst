Mujoco Model
============

The `MujocoModel` class is the foundational class from which all other model classes extend from in robosuite. This class represents a standardized API for all models used in simulation and is the core modeling component that other model classes build upon. The `MujocoXMLModel` is an extension of this class that represents models based on an XML file.

Base Mujoco Model
-----------------

.. autoclass:: robosuite.models.base.MujocoModel

  .. automethod:: correct_naming
  .. automethod:: set_site_visibility
  .. automethod:: exclude_from_prefixing
  .. autoproperty:: name
  .. autoproperty:: naming_prefix
  .. autoproperty:: root_body
  .. autoproperty:: bodies
  .. autoproperty:: joints
  .. autoproperty:: actuators
  .. autoproperty:: sites
  .. autoproperty:: sensors
  .. autoproperty:: contact_geoms
  .. autoproperty:: visual_geoms
  .. autoproperty:: important_geoms
  .. autoproperty:: important_sites
  .. autoproperty:: important_sensors
  .. autoproperty:: bottom_offset
  .. autoproperty:: top_offset
  .. autoproperty:: horizontal_radius


XML Mujoco Model
----------------

.. autoclass:: robosuite.models.base.MujocoXMLModel

  .. autoproperty:: base_offset
  .. autoproperty:: contact_geom_rgba