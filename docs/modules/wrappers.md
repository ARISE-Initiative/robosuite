# Wrappers

Wrappers offer additional features to an environment. Custom wrappers can be implemented for the purpose of collecting data, recording videos, and modifying environments. We provide some example wrapper implementations in this folder. All wrappers should inherit a base [Wrapper](wrapper.py) class.

To use a wrapper for the environment, import it and wrap it around the environment.

```python
import robosuite
from robosuite.wrappers import CustomWrapper

env = robosuite.make("SawyerLift")
env = CustomWrapper(env)
```

DataCollectionWrapper
---------------------
[DataCollectionWrapper](data_collection_wrapper.py) saves trajectory information to disk. This [demo script](../scripts/demo_collect_and_playback_data.py) illustrates how to collect the rollout trajectories and replay them in the simulation.

DemoSamplerWrapper
------------------
[DemoSamplerWrapper](demo_sampler_wrapper.py) loads demonstrations as a dataset of trajectories and randomly resets the start state of episodes along the demonstration trajectories based on a certain schedule. This functionality is useful for training RL agents and has been adopted in several prior work (see [references](../scripts/demo_learning_curriculum.py)). We provide a [demo script](../scripts/demo_learning_curriculum.py) to show how to configure the demo sampler to load demonstrations from files and use them to change the initial state distribution of episodes.

GymWrapper
----------
[GymWrapper](gym_wrapper.py) implements the standard methods in [OpenAI Gym](https://github.com/openai/gym), which allows popular RL libraries to run with our environments using the same APIs as Gym. This [demo script](../scripts/demo_gym_functionality.py) shows how to convert robosuite environments into Gym interfaces using this wrapper.

```bash
pip install gym
```

The main difference between the joint velocity action space and the end effector action space supported by this wrapper is that instead of supplying joint velocities per arm, a **delta position** vector and **delta quaternion** (xyzw) should be supplied per arm, where these correspond to the relative changes in position and rotation of the end effector from its current pose.
