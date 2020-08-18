Object Model
============

.. (TODO) Finish doc

Base Object Model
-----------------

.. autoclass:: robosuite.models.objects.objects.MujocoObject

  .. automethod:: __init__
  .. automethod:: get_bottom_offset
  .. automethod:: get_top_offset
  .. automethod:: get_horizontal_radius
  .. automethod:: get_collision
  .. automethod:: get_visual
  .. automethod:: get_site_attrib_template

.. autoclass:: robosuite.models.objects.objects.MujocoXMLObject

  .. automethod:: __init__

.. autoclass:: robosuite.models.objects.objects.MujocoGeneratedObject

  .. automethod:: __init__
  .. automethod:: sanity_check
  .. automethod:: get_collision_attrib_template
  .. automethod:: get_visual_attrib_template
  .. automethod:: append_material