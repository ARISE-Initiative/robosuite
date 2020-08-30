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

Sample scripts that showcase various features of **robosuite** are available at [scripts](scripts). The purpose of each script and usage instructions can be found at the beginning of each file.

## Building Your Own Environments
**robosuite** offers great flexibility in creating your own environments. A [task](modeling/task) typically involves the participation of a [robot](modeling/robot_model) with [grippers](modeling/gripper_model) as its end-effectors, an [arena](modeling/arena) (workspace), and [objects](modeling/object_model) that the robot interacts with. For a detailed overview of our design architecture, please check out the [Overview](modules/overview) page in Modules. Our Modeling APIs provide methods of composing these modularized elements into a scene, which can be loaded in MuJoCo for simulation. To build your own environments, we recommend you take a look at the [Environment classes](simulation/environment) which have used these APIs to define robotics environments and tasks and the [source code](https://github.com/Unknown-Initiative/robosuite-dev/tree/master/robosuite/environments) of our standardized environments.

<!-- You can also find detailed documentations about [creating a custom object](docs/creating_object.md) and [creating a custom environment](docs/creating_environment.md). -->
