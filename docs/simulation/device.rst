Device
======

Devices allow for direct real-time interfacing with the MuJoCo simulation. The currently supported devices are ``Keyboard``. ``SpaceMouse`` and ``MjGUI``.

Base Device
-----------

.. autoclass:: robosuite.devices.device.Device

  .. automethod:: start_control
  .. automethod:: get_controller_state


Keyboard Device
---------------

.. autoclass:: robosuite.devices.keyboard.Keyboard

  .. automethod:: get_controller_state
  .. automethod:: on_press
  .. automethod:: on_release
  .. automethod:: _display_controls


SpaceMouse Device
-----------------

.. autoclass:: robosuite.devices.spacemouse.SpaceMouse

  .. automethod:: get_controller_state
  .. automethod:: run
  .. automethod:: control
  .. automethod:: control_gripper
  .. automethod:: _display_controls

MjGUI Device
------------
.. autoclass:: robosuite.devices.mjgui.MjGUI

  .. automethod:: get_controller_state
  .. automethod:: input2action
  .. automethod:: _display_controls
