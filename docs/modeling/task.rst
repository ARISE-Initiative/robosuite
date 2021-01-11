Task
=====

The `Task` class is responsible for integrating a given `Arena`, `RobotModel`, and set of `MujocoObjects` into a single element
tree that is then parsed and converted into an `MjSim` object.

Base Task
-----------------

.. autoclass:: robosuite.models.tasks.task.Task

  .. automethod:: __init__
  .. automethod:: merge_robot
  .. automethod:: merge_arena
  .. automethod:: merge_objects