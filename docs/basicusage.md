# Basic Usage

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
    action = np.random.randn(*env.action_spec[0].shape)
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
````

This script above creates a simulated environment with the on-screen renderer, which is useful for visualization and qualitative evaluation. The `step()` function takes an `action` as input and returns a tuple of `(obs, reward, done, info)` where `obs` is an `OrderedDict` containing observations `[(name_string, np.array), ...]`, `reward` is the immediate reward obtained per step, `done` is a Boolean flag indicating if the episode has terminated and `info` is a dictionary which contains additional metadata.

Many other parameters can be configured for each environment. They provide functionalities such as headless rendering, getting pixel observations, changing camera settings, using reward shaping, and adding extra low-level observations. Please refer to [Environment](modules/environments) modules and the [Environment class](simulation/environment) APIs for further details.

Demo scripts that showcase various features of **robosuite** are available [here](demos). The purpose of each script and usage instructions can be found at the beginning of each file.

