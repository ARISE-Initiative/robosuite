# I/O Devices

Devices are used to read user input and collect human demonstrations. Demonstrations can be collected by either using a keyboard or a [SpaceNavigator 3D Mouse](https://www.3dconnexion.com/spacemouse_compact/en/) with the [collect_human_demonstrations](robosuite/scripts/collect_human_demonstrations.py) script. More generally, we support any interface that implements the [Device](device.py) abstract base class. In order to support your own custom device, simply subclass this base class and implement the required methods.

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

## SpaceNavigator 3D Mouse

We support the use of a  [SpaceNavigator 3D Mouse](https://www.3dconnexion.com/spacemouse_compact/en/) as well.

**SpaceNavigator 3D Mouse controls**

|          Control          |                Command                |
| :-----------------------: | :-----------------------------------: |
|       Right button        |           reset simulation            |
|    Left button (hold)     |             close gripper             |
|   Move mouse laterally    |  move arm horizontally in x-y plane   |
|   Move mouse vertically   |          move arm vertically          |
| Twist mouse about an axis | rotate arm about a corresponding axis |
|      ESC (keyboard)       |                 quit                  |
