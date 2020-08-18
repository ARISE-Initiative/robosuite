Environment
===========

Base Environment
----------------

.. autoclass:: robosuite.environments.base.MujocoEnv

  .. automethod:: __init__
  .. automethod:: initialize_time
  .. automethod:: reset
  .. automethod:: step
  .. automethod:: reward
  .. automethod:: render
  .. automethod:: observation_spec
  .. automethod:: action_spec
  .. automethod:: reset_from_xml_string
  .. automethod:: find_contacts
  .. automethod:: close