# Sensors

Sensors are an important aspect of **robosuite**, and encompass an agent's feedback from interaction with the environment. Mujoco provides low-level APIs to directly interface with raw simulation data, though we provide more a more realistic interface via the `Observable` class API to model obtained sensory information.

#### Mujoco-Native Sensors

The simulator generates virtual physical signals as response to a robot's interactions. Virtual signals include images, force-torque measurements (from a force-torque sensor like the one included by default in the wrist of all [Gripper models](../modeling/robot_model.html#gripper-model)), pressure signals (e.g. from a sensor on the robot's finger or on the environment), etc. Raw sensor information (except cameras and joint sensors) can be accessed via the function `get_sensor_measurement` provided the name of the sensor.

Joint sensors provide information about the state of each robot's joint including position and velocity. In MuJoCo these are not measured by sensors, but resolved and set by the simulator as the result of the actuation forces. Therefore, they are not accessed through the common `get_sensor_measurement` function but as properties of the [Robot simulation API](../simulation/robot), i.e., `_joint_positions` and `_joint_velocities`.

Cameras bundle a name to a set of properties to render images of the environment such as the pose and pointing direction, field of view, and resolution. Inheriting from MuJoCo, cameras are defined in the [robot](../modeling/robot_model) and [arena models](../modeling/arena) and can be attached to any body. Images, as they would be generated from the cameras, are not accessed through `get_sensor_measurement` but via the renderer (see below). In a common user pipeline, images are not queried directly; we specify one or several cameras we want to use images from when we create the environment, and the images are generated and appended automatically to the observation dictionary.

#### Observables

**robosuite** provides a realistic, customizable interface via the [Observable](../source/robosuite.utils.html#module-robosuite.utils.observables) class API. Observables model realistic sensor sampling, in which ground truth data is sampled (`sensor`), passed through a corrupting function (`corrupter`), and then finally passed through a filtering function (`filter`). Moreover, each observable has its own `sampling_rate` and `delayer` function which simulates sensor delay. While default values are used to instantiate each observable during environment creation, each of these components can be modified by the user at runtime using `env.modify_observable(...)` . Moreover, each observable is assigned a modality, and are grouped together in the returned observation dictionary during the `env.step()` call. For example, if an environment consists of camera observations and a single robot's proprioceptive observations, the observation dict structure might look as follows:

```python
{
    "frontview_image": np.array(...),    # this has modality "image"
    "frontview_depth": np.array(...),    # this has modality "image"
    "robot0_joint_pos": np.array(...),   # this has modality "robot0_proprio"
    "robot0_gripper_pos": np.array(...), # this has modality "robot0_proprio"
    "image-state": np.array(...),           # this is a concatenation of all image observations
    "robot0_proprio-state": np.array(...),  # this is a concatenation of all robot0_proprio observations
}
```

Note that for memory efficiency the `image-state` is not returned by default (this can be toggled in `robosuite/utils/macros.py`).

We showcase how the `Observable` functionality can be used to model sensor corruption and delay via [demo_sensor_corruption.py](../demos.html#sensor-realism). We also highlight that each of the `sensor`, `corrupter`, and `filter` functions can be arbitrarily specified to suit the end-user's usage. For example, a common use case for these observables is to keep track of sampled values from a sensor operating at a higher frequency than the environment step (control) frequency. In this case, the `filter` function can be leveraged to keep track of the real-time sensor values as they're being sampled. We provide a minimal script showcasing this ability below:

```python
import robosuite as suite
import numpy as np
from robosuite.utils.buffers import RingBuffer

# Create env instance
control_freq = 10
env = suite.make("Lift", robots="Panda", has_offscreen_renderer=False, use_camera_obs=False, control_freq=control_freq)

# Define a ringbuffer to store joint position values
buffer = RingBuffer(dim=env.robots[0].robot_model.dof, length=10)

# Create a function that we'll use as the "filter" for the joint position Observable
# This is a pass-through operation, but we record the value every time it gets called
# As per the Observables API, this should take in an arbitrary numeric and return the same type / shape
def filter_fcn(corrupted_value):
    # Record the inputted value
    buffer.push(corrupted_value)
    # Return this value (no-op performed)
    return corrupted_value

# Now, let's enable the joint position Observable with this filter function
env.modify_observable(
    observable_name="robot0_joint_pos",
    attribute="filter",
    modifier=filter_fcn,
)

# Let's also increase the sampling rate to showcase the Observable's ability to update multiple times per env step
obs_sampling_freq = control_freq * 4
env.modify_observable(
    observable_name="robot0_joint_pos",
    attribute="sampling_rate",
    modifier=obs_sampling_freq,
)

# Take a single environment step with positive joint velocity actions
arm_action = np.ones(env.robots[0].robot_model.dof) * 1.0
gripper_action = [1]
action = np.concatenate([arm_action, gripper_action])
env.step(action)

# Now we can analyze what values were recorded
np.set_printoptions(precision=2)
print(f"\nPolicy Frequency: {control_freq}, Observable Sampling Frequency: {obs_sampling_freq}")
print(f"Number of recorded samples after 1 policy step: {buffer._size}\n")
for i in range(buffer._size):
    print(f"Recorded value {i}: {buffer.buf[i]}")
```