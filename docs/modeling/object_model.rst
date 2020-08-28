Object Model
============

The MujocoObject class serves as a catch-all base class that is used to capture individual simulation objects to
instantiate within a given simulation. This is done in one of two ways -- the MujocoXMLObject reads in information from
a corresponding object XML file, whereas the MujocoGeneratedObject proecedurally generates a custom object using a
suite of utility mj modeling functions. In conjunction with the RobotModel, GripperModel, and Arena classes, these
classes serve as the basis for forming the higher level ManipulationTask class which is used to ultimately
generate the MjSim simulation object.

Base Object Model
-----------------

.. autoclass:: robosuite.models.objects.objects.MujocoObject

  .. automethod:: get_bottom_offset
  .. automethod:: get_top_offset
  .. automethod:: get_horizontal_radius
  .. automethod:: get_collision
  .. automethod:: get_visual
  .. automethod:: get_site_attrib_template


XML Object Model
-----------------

.. autoclass:: robosuite.models.objects.objects.MujocoXMLObject


Generated Object Model
-----------------

.. autoclass:: robosuite.models.objects.objects.MujocoGeneratedObject

  .. automethod:: sanity_check
  .. automethod:: get_collision_attrib_template
  .. automethod:: get_visual_attrib_template
  .. automethod:: append_material