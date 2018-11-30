# Surreal Robotics Suite

![gallery of_environments](resources/gallery.png)

Surreal Robotics Suite is a tookit and simulation benchmark powered by the [MuJoCo physics engine](http://mujoco.org/) for reproducible robotics research. The current release concentrates on reinforcement learning for robot manipulation. This library is designed to smoothly interoperate with the [Surreal Distributed Reinforcement Learning Framework](https://github.com/SurrealAI/Surreal).

Reinforcement learning has been a powerful and generic tool in robotics. Reinforcement learning combined with deep neural networks, i.e. *deep reinforcement learning* (DRL), has achieved some exciting successes in a variety of robot control problems. However, the challenges of reproducibility and replicability in DRL and robotics have impaired research progress. Our goal is to provide an accessible set of benchmarking tasks that facilitates a fair and rigorus evaluation and improves our understanding of new methods.

This framework was originally developed since late 2017 by researchers in [Stanford Vision and Learning Lab](http://svl.stanford.edu/) (SVL) as an internal tool for robot learning research. Today it is actively maintained and used for robotics research projects in SVL.

This release of Surreal Robotics Suite contains a set of benchmarking manipulation tasks and a modularized design of APIs for building new environments. We highlight these primary features below:

* [**standardized tasks**](robosuite/environments): a set of single-arm and bimanual manipulation tasks of large diversity and varying complexity.
* [**procedural generation**](robosuite/models): modularized APIs for programmatically creating new scenes and new tasks as a combinations of robot models, arenas, and parameterized 3D objects;
* [**controller modes**](robosuite/controllers): a selection of controller types to command the robots, such as joint velocity control, inverse kinematics control, and 3D motion devices for teleoperation;
* **multi-modal sensors**: heterogeneous types of sensory signals, including low-level physical states, RGB cameras, depth maps, and proprioception;
* [**human demonstrations**](docs/demonstrations.md): utilities for collecting human demonstrations, replaying demonstration datasets, and leveraging demonstration data for learning.

## Installation
Surreal Robotics Suite officially supports Mac OS X and Linux on Python 3.5 or 3.7. It can be run with an on-screen display for visualization or in a headless mode for model training, with or without a GPU.

The base installation requires the MuJoCo physics engine (with [mujoco-py](https://github.com/openai/mujoco-py), refer to link for troubleshooting the installation and further instructions) and [numpy](http://www.numpy.org/). To avoid interfering with system packages, it is recommended to install it under a virtual environment by first running `virtualenv -p python3 . && source bin/activate`.

First download MuJoCo 1.5.0 ([Linux](https://www.roboti.us/download/mjpro150_linux.zip) and [Mac OS X](https://www.roboti.us/download/mjpro150_osx.zip)) and place the `mjpro150` folder and your license key `mjkey.txt` in `~/.mujoco`. You can obtain a license key from [here](https://www.roboti.us/license.html).
   - For Linux, you will need to install some packages to build `mujoco-py` (sourced from [here](https://github.com/openai/mujoco-py/blob/master/Dockerfile), with a couple missing packages added). If using `apt`, the required installation command is:
     ```sh
     $ sudo apt install curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
             libosmesa6-dev software-properties-common net-tools unzip vim \
             virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf
     ```
     Note that for older versions of Ubuntu (e.g., 14.04) there's no libglfw3 package, in which case you need to `export LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin` before proceeding to the next step.

### Install from pip
1. After setting up mujoco, robosuite can be installed with
```sh
    $ pip install robosuite
```

2. Test your installation with
```sh
    $ python -m robosuite.demo
```

### Install from source
1. Clone the robosuite repository
```sh 
    $ git clone https://github.com/StanfordVL/robosuite.git
    $ cd robosuite
```

2. Install the base requirements with
   ```sh
   $ pip3 install -r requirements.txt
   ```
   This will also install our library as an editable package, such that local changes will be reflected elsewhere without having to reinstall the package.

3. (Optional) We also provide add-on functionalities, such as [OpenAI Gym](https://github.com/openai/gym) [interfaces](robosuite/wrappers/gym_wrapper.py), [inverse kinematics controllers](robosuite/wrappers/ik_wrapper.py) powered by [PyBullet](http://bulletphysics.org), and [teleoperation](robosuite/scripts/demo_spacemouse_ik_control.py) with [SpaceMouse](https://www.3dconnexion.com/products/spacemouse.html) devices (Mac OS X only). To enable these additional features, please install the extra dependencies by running
   ```sh
   $ pip3 install -r requirements-extra.txt
   ```

4. Test your installation with
```sh
    $ python robosuite/demo.py
```

## Quick Start
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

## Human Demonstrations

### Collecting Human Demonstrations

We provide teleoperation utilities that allow users to control the robots with input devices, such as the keyboard and the [SpaceMouse](https://www.3dconnexion.com/spacemouse_compact/en/). Such functionality allows us to collect a dataset of human demonstrations for learning. We provide an example script to illustrate how to collect demonstrations. Our [collect_human_demonstrations](robosuite/scripts/collect_human_demonstrations.py) script takes the following arguments:

- `directory:` path to a folder for where to store the pickle file of collected demonstrations
- `environment:` name of the environment you would like to collect the demonstrations for
- `device:` either "keyboard" or "spacemouse"

Our twin project [RoboTurk](http://roboturk.stanford.edu) has collected pilot datasets of more than a thousand demonstrations for two tasks in our Suite via crowdsourcing. You can find detailed information about the [RoboTurk datasets](docs/demonstrations.md#roboturk-dataset) and [demonstration collection](docs/demonstrations.md#collecting-your-own-demonstrations) here.

### Replaying Human Demonstrations

We have included an example script that illustrates how demonstrations can be loaded and played back. Our [playback_demonstrations_from_hdf5](robosuite/scripts/playback_demonstrations_from_hdf5.py) script selects demonstration episodes at random from a demonstration pickle file and replays them. We have included some sample demonstrations for each task at `models/assets/demonstrations`.

### Using Demonstrations for Learning

[Several](https://arxiv.org/abs/1802.09564) [prior](https://arxiv.org/abs/1807.06919) [works](https://arxiv.org/abs/1804.02717) have demonstrated the effectiveness of altering the start state distribution of training episodes for learning RL policies. We provide a generic utility for setting various types of learning curriculums which dictate how to sample from demonstration episodes when doing an environment reset. For more information see the `DemoSamplerWrapper` class. We have provided an example of how to use this wrapper along with a demonstration pickle file in the [demo_learning_curriculum](robosuite/scripts/demo_learning_curriculum.py) script.

## Citations
Please cite [Surreal](http://surreal.stanford.edu) if you use this repository in your publications:
```
@inproceedings{corl2018surreal,
  title={SURREAL: Open-Source Reinforcement Learning Framework and Robot Manipulation Benchmark},
  author={Fan, Linxi and Zhu, Yuke and Zhu, Jiren and Liu, Zihua and Zeng, Orien and Gupta, Anchit and Creus-Costa, Joan and Savarese, Silvio and Fei-Fei, Li},
  booktitle={Conference on Robot Learning},
  year={2018}
}
```

Please also cite [RoboTurk](http://roboturk.stanford.edu) if you use the demonstration datasets:
```
@inproceedings{corl2018roboturk,
  title={RoboTurk: A Crowdsourcing Platform for Robotic Skill Learning through Imitation},
  author={Mandlekar, Ajay and Zhu, Yuke and Garg, Animesh and Booher, Jonathan and Spero, Max and Tung, Albert and Gao, Julian and Emmons, John and Gupta, Anchit and Orbay, Emre and Savarese, Silvio and Fei-Fei, Li},
  booktitle={Conference on Robot Learning},
  year={2018}
}
```
