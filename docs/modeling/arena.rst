Arena
=====

The ``Arena`` class serves as a base model for building the simulation environment.
By default, this includes a ground plane and visual walls, and child classes extend this
to additionally include other objects, e.g., a table or bins.

Base Arena
----------

.. autoclass:: robosuite.models.arenas.arena.Arena

  .. automethod:: __init__
  .. automethod:: set_origin
  .. automethod:: set_camera

Empty Arena
-----------

.. autoclass:: robosuite.models.arenas.empty_arena.EmptyArena

  .. automethod:: __init__

Bins Arena
----------

.. autoclass:: robosuite.models.arenas.bins_arena.BinsArena

  .. automethod:: __init__
  .. automethod:: configure_location

Pegs Arena
----------

.. autoclass:: robosuite.models.arenas.pegs_arena.PegsArena

  .. automethod:: __init__

Table Arena
-----------

.. autoclass:: robosuite.models.arenas.table_arena.TableArena

  .. automethod:: __init__
  .. automethod:: configure_location
  .. autoproperty:: table_top_abs

Wipe Arena
----------

.. autoclass:: robosuite.models.arenas.wipe_arena.WipeArena

  .. automethod:: __init__
  .. automethod:: configure_location
  .. automethod:: reset_arena
  .. automethod:: sample_start_pos
  .. automethod:: sample_path_pos

MultiTable Arena
----------------

.. autoclass:: robosuite.models.arenas.multi_table_arena.MultiTableArena

  .. automethod:: __init__
  .. automethod:: _add_table
  .. automethod:: configure_location
  .. automethod:: _postprocess_arena
