Object Model
============

The `MujocoObject` class serves as a catch-all base class that is used to capture individual simulation objects to
instantiate within a given simulation. This is done in one of two ways via extended classes -- the `MujocoXMLObject`
reads in information from a corresponding object XML file, whereas the `MujocoGeneratedObject` proecedurally generates a
custom object using a suite of utility mj modeling functions. In conjunction with the `RobotModel`, and
`Arena` classes, these classes serve as the basis for forming the higher level `Task` class which is used to
ultimately generate the `MjSim` simulation object.

Base Object Model
-----------------

.. autoclass:: robosuite.models.objects.objects.MujocoObject

  .. automethod:: __init__
  .. automethod:: merge_assets
  .. automethod:: get_obj
  .. automethod:: exclude_from_prefixing
  .. automethod:: _get_object_subtree
  .. automethod:: _get_object_properties
  .. autoproperty:: important_geoms
  .. autoproperty:: important_sites
  .. autoproperty:: important_sensors
  .. autoproperty:: get_site_attrib_template
  .. autoproperty:: get_joint_attrib_template


XML Object Model
----------------

.. autoclass:: robosuite.models.objects.objects.MujocoXMLObject

  .. automethod:: __init__
  .. automethod:: _duplicate_visual_from_collision
  .. automethod:: _get_geoms


Generated Object Model
----------------------

.. autoclass:: robosuite.models.objects.objects.MujocoGeneratedObject

  .. automethod:: __init__
  .. automethod:: sanity_check
  .. automethod:: get_collision_attrib_template
  .. automethod:: get_visual_attrib_template
  .. automethod:: append_material