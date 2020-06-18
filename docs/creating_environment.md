# How to build a custom environment
We provide a variety of templating tools to build an environment in a modular way. Here we break down the creation of `SawyerLift` to demonstrate these functionalities. The code cited ([here](../robosuite/environments/sawyer_lift.py#L138)) can be found in `_load_model` methods of classes `SawyerEnv` (creates the robot) and `SawyerLift` (creates the table) plus the code of `TableTopTask`.

# Modeling
Here we explain step-by-step how to create a model of a manipulation task using our APIs.

## Creating the world
All mujoco object definitions are housed in an xml. We create a `MujocoWorldBase` class to do it.
```python
from robosuite.models import MujocoWorldBase

world = MujocoWorldBase()
```

## Creating the robot
The class housing the xml of a robot can be created as follows
```python
from robosuite.models.robots import Sawyer

mujoco_robot = Sawyer()
```

# TODO: Update!

We can add a gripper to the robot by creating a gripper instance and calling the `add_gripper` method on a robot
```python
from robosuite.models.grippers import gripper_factory

gripper = gripper_factory('RethinkGripper')
gripper.hide_visualization()
mujoco_robot.add_gripper("right_hand", gripper)
```

To add the robot to the world, we place the robot on to a desired position and merge it into the world
```python
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)
```

## Creating the table
We can initialize the `TableArena` instance that creates a table and the floorplane
```python
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.16, 0, 0])
world.merge(mujoco_arena)
```

## Adding the object
For details of `MujocoObject`, refer to the [documentation about MujocoObject](objects.md), we can create a ball and add it to the world. It is a bit more complicated than before because we are adding a free joint to the object (so it can move) and we want to place the object properly
```python
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import new_joint

object_mjcf = BoxObject()
world.merge_asset(object_mjcf)

obj = object_mjcf.get_collision(name="box_object", site=True)
obj.append(new_joint(name="box_object", type="free"))
obj.set("pos", [0, 0, 0.5])
world.worldbody.append(obj)
```

# Simulation
Once we have created the object, we can obtain a [mujoco_py](https://github.com/openai/mujoco-py) model by running
```python
model = world.get_model(mode="mujoco_py")
```
This is a mujoco_py `MjModel` instance than can then be used for simulation.

For example, 
```python
from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
sim.data.ctrl[:] = [1,2,3,4,5]
sim.step()
viewer.render()
```

For details, refer to [mujoco_py](https://github.com/openai/mujoco-py)'s documentation or look at our implementations in the environments.
