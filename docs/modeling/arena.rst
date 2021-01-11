Arena
=====

The `Arena` class serves as a base model for building the simulation environment.
By default, this includes a ground plane and visual walls, and child classes extend this
to additionally include other objects, e.g.: a table or bins.

Base Arena
----------

.. autoclass:: robosuite.models.arenas.arena.Arena

  .. automethod:: __init__
  .. automethod:: set_origin
  .. automethod:: set_camera
