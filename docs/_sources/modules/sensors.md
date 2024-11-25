# Sensors

Sensors are an important aspect of **robosuite**, and encompass an agent's feedback from interaction with the environment. Mujoco provides low-level APIs to directly interface with raw simulation data, though we provide more a more realistic interface via the `Observable` class API to model obtained sensory information.

#### Mujoco-Native Sensors

The simulator generates virtual physical signals as response to a robot's interactions. Virtual signals include images, force-torque measurements (from a force-torque sensor like the one included by default in the wrist of all [Gripper models](../modeling/robot_model.html#gripper-model)), pressure signals (e.g. from a sensor on the robot's finger or on the environment), etc. Raw sensor information (except cameras and joint sensors) can be accessed via the function `get_sensor_measurement` provided the name of the sensor.

Joint sensors provide information about the state of each robot's joint including position and velocity. In MuJoCo these are not measured by sensors, but resolved and set by the simulator as the result of the actuation forces. Therefore, they are not accessed through the common `get_sensor_measurement` function but as properties of the [Robot simulation API](../simulation/robot), i.e., `_joint_positions` and `_joint_velocities`.

Cameras bundle a name to a set of properties to render images of the environment such as the pose and pointing direction, field of view, and resolution. Inheriting from MuJoCo, cameras are defined in the [robot](../modeling/robot_model) and [arena models](../modeling/arena) and can be attached to any body. Images, as they would be generated from the cameras, are not accessed through `get_sensor_measurement` but via the renderer (see below). In a common user pipeline, images are not queried directly; we specify one or several cameras we want to use images from when we create the environment, and the images are generated and appended automatically to the observation dictionary.

#### Observables

**robosuite** provides a realistic, customizable interface via the [Observable](../source/robosuite.utils.html#robosuite.utils.observables.Observable) class API. Observables model realistic sensor sampling, in which ground truth data is sampled (`sensor`), passed through a corrupting function (`corrupter`), and then finally passed through a filtering function (`filter`). Moreover, each observable has its own `sampling_rate` and `delayer` function which simulates sensor delay. While default values are used to instantiate each observable during environment creation, each of these components can be modified by the user at runtime using `env.modify_observable(...)` . Moreover, each observable is assigned a modality, and are grouped together in the returned observation dictionary during the `env.step()` call. For example, if an environment consists of camera observations (RGB, depth, and instance segmentation) and a single robot's proprioceptive observations, the observation dict structure might look as follows:

```python
{
    "frontview_image": np.array(...),                   # this has modality "image"
    "frontview_depth": np.array(...),                   # this has modality "image"
    "frontview_segmentation_instance": np.array(...),   # this has modality "image"
    "robot0_joint_pos": np.array(...),                  # this has modality "robot0_proprio"
    "robot0_gripper_pos": np.array(...),                # this has modality "robot0_proprio"
    "image-state": np.array(...),                       # this is a concatenation of all image observations
    "robot0_proprio-state": np.array(...),              # this is a concatenation of all robot0_proprio observations
}
```

For more information on the vision ground-truth sensors supported, please see the [Renderer](./renderers) section.

Note that for memory efficiency the `image-state` is not returned by default (this can be toggled in `robosuite/macros.py`).

Observables can also be used to model sensor corruption and delay, and refer the reader to the [Sensor Randomization](../algorithms/sim2real.html#sensors) section for additional information.
