Environment
===========

The MujocoEnv class defines a top-level simulation object encapsulating a MjSim object. Robosuite uses
class extensions of this base class, namely, RobotEnv which additionally encompasses Robot objects and the top-level
task environments which capture specific ManipulationTask instances and additional objects.

Base Environment
----------------

.. autoclass:: robosuite.environments.base.MujocoEnv

  .. automethod:: initialize_time
  .. automethod:: reset
  .. automethod:: step
  .. automethod:: reward
  .. automethod:: render
  .. automethod:: observation_spec
  .. autoproperty:: action_spec
  .. autoproperty:: action_dim
  .. automethod:: reset_from_xml_string
  .. automethod:: find_contacts
  .. automethod:: close


Robot Environment
-----------------

.. autoclass:: robosuite.environments.robot_env.RobotEnv

  .. automethod:: move_indicator
  .. automethod:: _pre_action
  .. automethod:: _post_action
  .. automethod:: _get_observation
  .. automethod:: _check_gripper_contact
  .. automethod:: _check_arm_contact
  .. automethod:: _check_q_limits
  .. automethod:: _visualization
  .. automethod:: _load_robots
  .. automethod:: _check_robot_configuration