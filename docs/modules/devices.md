# I/O Devices

Devices are used to read user input and teleoperate simulated robots in real-time. This is achieved by either using a keyboard or a [SpaceMouse](https://www.3dconnexion.com/spacemouse_compact/en/), and whose teleoperation capabilities can be demonstrated with the [demo_device_control.py](../demos.html#teleoperation) script. More generally, we support any interface that implements the [Device](../simulation/device) abstract base class. In order to support your own custom device, simply subclass this base class and implement the required methods.

## Keyboard

We support keyboard input through the OpenCV2 window created by the mujoco renderer. 

**Keyboard controls**

Note that the rendering window must be active for these commands to work.

|        Keys         |                   Command                  |
| :------------------ | :----------------------------------------- |
|      Ctrl+q         |               reset simulation             |
|     spacebar        |          toggle gripper (open/close)       |
| up-right-down-left  |       move horizontally in x-y plane       |
|        .-;          |                move vertically             |
|        o-p          |                 rotate (yaw)               |
|        y-h          |                rotate (pitch)              |
|        e-r          |                 rotate (roll)              |
|         b           |     toggle arm/base mode (if appli cable)  |
|         s           |  switch active arm (if multi-armed robot)  |
|         =           | switch active robot (if multi-robot env)   |
|        ESC          |                    quit                    |


## 3Dconnexion SpaceMouse

We support the use of a [SpaceMouse](https://www.3dconnexion.com/spacemouse_compact/en/) as well.

**3Dconnexion SpaceMouse controls**

|          Control          |                Command                |
| :------------------------ | :------------------------------------ |
|       Right button        |           reset simulation            |
|    Left button (hold)     |             close gripper             |
|   Move mouse laterally    |  move arm horizontally in x-y plane   |
|   Move mouse vertically   |          move arm vertically          |
| Twist mouse about an axis | rotate arm about a corresponding axis |
|      ESC (keyboard)       |                 quit                  |


## Mujoco GUI Device

To use the Mujoco GUI device for teleoperation, follow these steps:

1. Set renderer as `"mjviewer"`. For example:

```python
env = suite.make(
    **options,
    renderer="mjviewer",
    has_renderer=True,
    has_offscreen_renderer=False,
    ignore_done=True,
    use_camera_obs=False,
)
```

Note: if using Mac, please use `mjpython` instead of `python`. For example:

```mjpython robosuite/scripts/collect_human_demonstrations.py --environment Lift --robots Panda --device mjgui --camera frontview --controller WHOLE_BODY_IK```

2. Double click on a mocap body to select a body to drag, then:

On Linux: `Ctrl` + right click to drag the body's position. `Ctrl` + left click to control the body's orientation.
On Mac: `fn` + `Ctrl` + right click.
