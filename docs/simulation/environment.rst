Environment
===========

The ``MujocoEnv`` class defines a top-level simulation object encapsulating a ``MjSim`` object. Robosuite uses
class extensions of this base class, namely, ``RobotEnv`` which additionally encompasses ``Robot`` objects and the top-level
task environments which capture specific ``ManipulationTask`` instances and additional objects.

Base Environment
----------------

.. autoclass:: robosuite.environments.base.MujocoEnv

  .. automethod:: initialize_time
  .. automethod:: set_model_postprocessor
  .. automethod:: reset
  .. automethod:: step
  .. automethod:: reward
  .. automethod:: render
  .. automethod:: observation_spec
  .. automethod:: clear_objects
  .. automethod:: visualize
  .. automethod:: reset_from_xml_string
  .. automethod:: check_contact
  .. automethod:: get_contacts
  .. automethod:: modify_observable
  .. automethod:: close
  .. autoproperty:: observation_modalities
  .. autoproperty:: observation_names
  .. autoproperty:: enabled_observables
  .. autoproperty:: active_observables
  .. autoproperty:: action_spec
  .. autoproperty:: action_dim


Robot Environment
-----------------

.. autoclass:: robosuite.environments.robot_env.RobotEnv

  .. automethod:: _load_robots
  .. automethod:: _check_robot_configuration


Manipulator Environment
-----------------------

.. autoclass:: robosuite.environments.manipulation.manipulation_env.ManipulationEnv

  .. automethod:: _check_grasp
  .. automethod:: _gripper_to_target
  .. automethod:: _visualize_gripper_to_target