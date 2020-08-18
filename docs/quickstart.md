# Quick Start

## Creating Environments
The APIs we provide to interact with our environments are simple and similar to the ones used by [OpenAI Gym](https://github.com/openai/gym/). Below is a minimalistic example of how to interact with an environment.

```python
import numpy as np
import robosuite as suite

# create environment instance
env = suite.make("SawyerLift", has_renderer=True)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.dof)  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
````
The `step()` function takes an `action` as input and returns a tuple of `(obs, reward, done, info)` where `obs` is an `OrderedDict` containing observations `[(name_string, np.array), ...]`, `reward` is the immediate reward obtained per step, `done` is a Boolean flag indicating if the episode has terminated and `info` is a dictionary which contains additional metadata.

There are other parameters which can be configured for each environment. They provide functionalities such as headless rendering, getting pixel observations, changing camera settings, using reward shaping, and adding extra low-level observations. Please refer to [this page](robosuite/environments/README.md) and the [environment classes](robosuite/environments) for further details.

Sample scripts that showcase various features of the Surreal Robotics Suite are available at [robosuite/scripts](robosuite/scripts). The purpose of each script and usage instructions can be found at the beginning of each file.

## Building Your Own Environments
A manipulation `task` typically involves the participation of a `robot` with `gripper`s as its end-effectors, an `arena` (workspace), and `object`s that the robot interacts with. Our APIs in [Models](robosuite/models) provide a toolkit of composing these modularized elements into a scene, which can be loaded in MuJoCo for simulation. To build your own environments, you are recommended to take a look at the [environment classes](robosuite/environments) which have used these APIs to define a set of standardized manipulation tasks. You can also find detailed documentations about [creating a custom object](docs/creating_object.md) and [creating a custom environment](docs/creating_environment.md).