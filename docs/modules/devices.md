# I/O Devices

Devices are used to read user input and teleoperate simulated robots in real time. This is achieved by either using a keyboard or a [SpaceMouse](https://www.3dconnexion.com/spacemouse_compact/en/), and whose teleoperative capabilities can be demonstrated with the [demo_device_control.py](../demos.html#teleoperation) script. More generally, we support any interface that implements the [Device](../simulation/device) abstract base class. In order to support your own custom device, simply subclass this base class and implement the required methods.

## Keyboard

We support keyboard input through the GLFW window created by the mujoco-py renderer. 

**Keyboard controls**

Note that the rendering window must be active for these commands to work.

|   Keys   |              Command               |
| :------: | :--------------------------------: |
|    q     |          reset simulation          |
| spacebar |    toggle gripper (open/close)     |
| w-a-s-d  | move arm horizontally in x-y plane |
|   r-f    |        move arm vertically         |
|   z-x    |      rotate arm about x-axis       |
|   t-g    |      rotate arm about y-axis       |
|   c-v    |      rotate arm about z-axis       |
|   ESC    |                quit                |

## 3Dconnexion SpaceMouse

We support the use of a [SpaceMouse](https://www.3dconnexion.com/spacemouse_compact/en/) as well.

**3Dconnexion SpaceMouse controls**

|          Control          |                Command                |
| :-----------------------: | :-----------------------------------: |
|       Right button        |           reset simulation            |
|    Left button (hold)     |             close gripper             |
|   Move mouse laterally    |  move arm horizontally in x-y plane   |
|   Move mouse vertically   |          move arm vertically          |
| Twist mouse about an axis | rotate arm about a corresponding axis |
|      ESC (keyboard)       |                 quit                  |
