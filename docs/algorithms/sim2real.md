# Sim-to-Real Transfer
This page covers the randomization techniques to narrow the reality gap of our robotics simulation. These techniques, which concerns about [visual observations](#visuals), [system dynamics](#dynamics), and [sensors](#sensors), are employed to improve the efficacy of transferring our simulation-trained models to the real world.


## Visuals

It is well shown that randomizing the visuals in simulation can play an important role in sim2real applications. **robosuite** provides various `Modder` classes to control different aspects of the visual environment. This includes:

- `CameraModder`: Modder for controlling camera parameters, including FOV and pose
- `TextureModder`: Modder for controlling visual objects' appearances, including texture and material properties
- `LightingModder`: Modder for controlling lighting parameters, including light source properties and pose

Each of these Modders can be used by the user to directly override default simulation settings, or to randomize their respective properties mid-sim. We provide [demo_domain_randomization.py](../demos.html#domain-randomization) to showcase all of these modders being applied to randomize an environment during every sim step.


## Dynamics

In order to achieve reasonable runtime speeds, many physics simulation platforms often must simplify the underlying physics model. Mujoco is no different, and as a result, many parameters such as friction, damping, and contact constraints do not fully capture real-world dynamics.

To better compensate for this, **robosuite** provides the `DynamicsModder` class, which can control individual dynamics parameters for each model within an environment. Theses parameters are sorted by element group, and briefly described below (for more information, please see [Mujoco XML Reference](http://www.mujoco.org/book/XMLreference.html)):
 
#### Opt (Global) Parameters
- `density`: Density of the medium (i.e.: air)
- `viscosity`: Viscosity of the medium (i.e.: air)

#### Body Parameters
- `position`: (x, y, z) Position of the body relative to its parent body
- `quaternion`: (qw, qx, qy, qz) Quaternion of the body relative to its parent body
- `inertia`: (ixx, iyy, izz) diagonal components of the inertia matrix associated with this body
- `mass`: mass of the body

#### Geom Parameters
- `friction`: (sliding, torsional, rolling) friction values for this geom
- `solref`: (timeconst, dampratio) contact solver values for this geom
- `solimp`: (dmin, dmax, width, midpoint, power) contact solver impedance values for this geom

#### Joint parameters
- `stiffness`: Stiffness for this joint
- `frictionloss`: Friction loss associated with this joint
- `damping`: Damping value for this joint
- `armature`: Gear inertia for this joint

This `DynamicsModder` follows the same basic API as the other `Modder` classes, and allows per-parameter and per-group randomization enabling. Apart from randomization, this modder can also be instantiated to selectively modify values at runtime. A brief example is given below:

```python
import robosuite as suite
from robosuite.utils.mjmod import DynamicsModder
import numpy as np

# Create environment and modder
env = suite.make("Lift", robots="Panda")
modder = DynamicsModder(sim=env.sim, random_state=np.random.RandomState(5))

# Define function for easy printing
cube_body_id = env.sim.model.body_name2id(env.cube.root_body)
cube_geom_ids = [env.sim.model.geom_name2id(geom) for geom in env.cube.contact_geoms]

def print_params():
    print(f"cube mass: {env.sim.model.body_mass[cube_body_id]}")
    print(f"cube frictions: {env.sim.model.geom_friction[cube_geom_ids]}")
    print()

# Print out initial parameter values
print("INITIAL VALUES")
print_params()

# Modify the cube's properties
modder.mod(env.cube.root_body, "mass", 5.0)                                # make the cube really heavy
for geom_name in env.cube.contact_geoms:
    modder.mod(geom_name, "friction", [2.0, 0.2, 0.04])           # greatly increase the friction
modder.update()                                                   # make sure the changes propagate in sim

# Print out modified parameter values
print("MODIFIED VALUES")
print_params()

# We can also restore defaults (original values) at any time
modder.restore_defaults()

# Print out restored initial parameter values
print("RESTORED VALUES")
print_params()
```

Running [demo_domain_randomization.py](../demos.html#domain-randomization) is another method for demo'ing (albeit an extreme example of) this functionality.

Note that the modder already has some sanity checks in place to prevent presumably undesired / non-sensical behavior, such as adding damping / frictionloss to a free joint or setting a non-zero stiffness value to a joint that is normally non-stiff to begin with.


## Sensors

By default, Mujoco sensors are deterministic and delay-free, which is often an unrealistic assumption to make in the real world. To better close this domain gap, **robosuite** provides a realistic, customizable interface via the [Observable](../source/robosuite.utils.html#module-robosuite.utils.observables) class API. Observables model realistic sensor sampling, in which ground truth data is sampled (`sensor`), passed through a corrupting function (`corrupter`), and then finally passed through a filtering function (`filter`). Moreover, each observable has its own `sampling_rate` and `delayer` function which simulates sensor delay. While default values are used to instantiate each observable during environment creation, each of these components can be modified by the user at runtime using `env.modify_observable(...)` . Moreover, each observable is assigned a modality, and are grouped together in the returned observation dictionary during the `env.step()` call. For example, if an environment consists of camera observations and a single robot's proprioceptive observations, the observation dict structure might look as follows:

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