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

**Step 4: Adding the object.** For details of `MujocoObject`, refer to the documentation about [MujocoObject](modeling/object_model), we can create a ball and add it to the world.
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
model = world.get_model(mode="mujoco")
```
This is an `MjModel` instance that can then be used for simulation. For example,
```python
import mujoco

data = mujoco.MjData(model)
while data.time < 1:
    mujoco.mj_step(model, data)
```

