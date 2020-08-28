Task
=====

The Task class is responsible for integrating a given Arena, RobotModel, and set of MujocoObjects into a single element
tree that is then parsed and converted into an MjSim object. It is also responsible for placing the objects within
the simulation.

Manipulation Task
-----------------

.. autoclass:: robosuite.models.tasks.ManipulationTask

  .. automethod:: merge_robot
  .. automethod:: merge_arena
  .. automethod:: merge_objects
  .. automethod:: place_objects
