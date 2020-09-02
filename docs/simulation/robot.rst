Robot
=====

The Robot class defines a simulation object encapsulating a robot model, gripper model, and controller. Robosuite uses
class extensions of this base class, namely, SingleArm and Bimanual classes representing the two different types of
supported robots.

Base Robot
----------

.. autoclass:: robosuite.robots.robot.Robot

  .. automethod:: load_model
  .. automethod:: reset_sim
  .. automethod:: reset
  .. automethod:: setup_references
  .. automethod:: control


SingleArm Robot
---------------

.. autoclass:: robosuite.robots.single_arm.SingleArm

  .. automethod:: control
  .. automethod:: grip_action


Bimanual Robot
--------------

.. autoclass:: robosuite.robots.bimanual.Bimanual

  .. automethod:: control
  .. automethod:: grip_action
