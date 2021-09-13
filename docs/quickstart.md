# Quick Start

## Running Standardized Environments
**robosuite** offers a set of standardized manipulation tasks for benchmarking purposes. These pre-defined environments can be easily instantiated with the `make` function. The APIs we provide to interact with our environments are simple and similar to the ones used by [OpenAI Gym](https://github.com/openai/gym/). Below is a minimalistic example of how to interact with an environment.

```python
import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
````

This script above creates a simulated environment with the on-screen renderer, which is useful for visualization and qualitative evaluation. The `step()` function takes an `action` as input and returns a tuple of `(obs, reward, done, info)` where `obs` is an `OrderedDict` containing observations `[(name_string, np.array), ...]`, `reward` is the immediate reward obtained per step, `done` is a Boolean flag indicating if the episode has terminated and `info` is a dictionary which contains additional metadata.

Many other parameters can be configured for each environment. They provide functionalities such as headless rendering, getting pixel observations, changing camera settings, using reward shaping, and adding extra low-level observations. Please refer to [Environment](modules/environments) modules and the [Environment class](simulation/environment) APIs for further details.

Demo scripts that showcase various features of **robosuite** are available [here](demos). The purpose of each script and usage instructions can be found at the beginning of each file.

## Building Your Own Environments
**robosuite** offers great flexibility in creating your own environments. A [task](modeling/task) typically involves the participation of a [robot](modeling/robot_model) with [grippers](modeling/robot_model.html#gripper-model) as its end-effectors, an [arena](modeling/arena) (workspace), and [objects](modeling/object_model) that the robot interacts with. For a detailed overview of our design architecture, please check out the [Overview](modules/overview) page in Modules. Our Modeling APIs provide methods of composing these modularized elements into a scene, which can be loaded in MuJoCo for simulation. To build your own environments, we recommend you take a look at the [Environment classes](simulation/environment) which have used these APIs to define robotics environments and tasks and the [source code](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments) of our standardized environments. Below we walk through a step-by-step example of building a new tabletop manipulation environment with our APIs.

**Step 1: Creating the world.** All mujoco object definitions are housed in an xml. We create a [MujocoWorldBase](source/robosuite.models) class to do it.
```python
from robosuite.models import MujocoWorldBase

world = MujocoWorldBase()
```

**Step 2: Creating the robot.** The class housing the xml of a robot can be created as follows.
```python
from robosuite.models.robots import Panda

mujoco_robot = Panda()
```
We can add a gripper to the robot by creating a gripper instance and calling the add_gripper method on a robot.
```python
from robosuite.models.grippers import gripper_factory

gripper = gripper_factory('PandaGripper')
mujoco_robot.add_gripper(gripper)
```
To add the robot to the world, we place the robot on to a desired position and merge it into the world
```python
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)
```

**Step 3: Creating the table.** We can initialize the [TableArena](source/robosuite.models.arenas) instance that creates a table and the floorplane
```python
from robosuite.models.arenas import TableArena

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)
```

**Step 4: Adding the object.** For details of `MujocoObject`, refer to the documentations about [MujocoObject](modeling/object_model), we can create a ball and add it to the world.
```python
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint

sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere)
```

**Step 5: Running Simulation.** Once we have created the object, we can obtain a `mujoco_py` model by running
```python
model = world.get_model(mode="mujoco_py")
```
This is an `MjModel` instance that can then be used for simulation. For example,
```
from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
  sim.data.ctrl[:] = 0
  sim.step()
  viewer.render()
```
