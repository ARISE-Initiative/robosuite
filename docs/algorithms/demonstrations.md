# Human Demonstrations

## Collecting Human Demonstrations

We provide teleoperation utilities that allow users to control the robots with input devices, such as the keyboard and the [SpaceMouse](https://www.3dconnexion.com/spacemouse_compact/en/). Such functionality allows us to collect a dataset of human demonstrations for learning. We provide an example script to illustrate how to collect demonstrations. Our [collect_human_demonstrations](robosuite/scripts/collect_human_demonstrations.py) script takes the following arguments:

- `directory:` path to a folder for where to store the pickle file of collected demonstrations
- `environment:` name of the environment you would like to collect the demonstrations for
- `device:` either "keyboard" or "spacemouse"

### Keyboard controls

Note that the rendering window must be active for these commands to work.

|   Keys   |              Command               |
| :------: | :--------------------------------: |
|    q     |          reset simulation          |
| spacebar |    toggle gripper (open/close)     |
| w-a-s-d  | move arm horizontally in x-y plane |
|   r-f    |        move arm vertically         |
|   z-x    |      rotate arm about x-axis       |
|   t-g    |      rotate arm about y-axis       |
|   c-v    |      rotate arm about z-axis       |
|   ESC    |                quit                |

### 3Dconnexion SpaceMouse controls

|          Control          |                Command                |
| :-----------------------: | :-----------------------------------: |
|       Right button        |           reset simulation            |
|    Left button (hold)     |             close gripper             |
|   Move mouse laterally    |  move arm horizontally in x-y plane   |
|   Move mouse vertically   |          move arm vertically          |
| Twist mouse about an axis | rotate arm about a corresponding axis |
|      ESC (keyboard)       |                 quit                  |



## Replaying Human Demonstrations

We have included an example script that illustrates how demonstrations can be loaded and played back. Our [playback_demonstrations_from_hdf5](robosuite/scripts/playback_demonstrations_from_hdf5.py) script selects demonstration episodes at random from a demonstration pickle file and replays them.


## Existing Datasets

We have included some sample demonstrations for each task at `models/assets/demonstrations`.

Our twin project [RoboTurk](http://roboturk.stanford.edu) has also collected pilot datasets of more than a thousand demonstrations for two tasks in our suite via crowdsourcing. You can find detailed information about the RoboTurk datasets [here](docs/demonstrations.md#roboturk-dataset). 


## Structure of collected demonstrations

Every set of demonstrations is collected as a `demo.hdf5` file. The `demo.hdf5` file is structured as follows.

- data (group)

  - date (attribute) - date of collection

  - time (attribute) - time of collection

  - repository_version (attribute) - repository version used during collection

  - env (attribute) - environment name on which demos were collected

  - demo1 (group) - group for the first demonstration (every demonstration has a group)

    - model_file (attribute) - the xml string corresponding to the MJCF mujoco model

    - states (dataset) - flattened mujoco states, ordered by time

    - actions (dataset) - environment actions, ordered by time

  - demo2 (group) - group for the second demonstration

    ... 

    (and so on)

The reason for storing mujoco states instead of raw observations is to make it easy to retrieve different kinds of observations in a postprocessing step. This also saves disk space (image datasets are much larger).


## Using Demonstrations for Learning

[Several](https://arxiv.org/abs/1802.09564) [prior](https://arxiv.org/abs/1807.06919) [works](https://arxiv.org/abs/1804.02717) have demonstrated the effectiveness of altering the start state distribution of training episodes for learning RL policies. We provide a generic utility for setting various types of learning curriculums which dictate how to sample from demonstration episodes when doing an environment reset. For more information see the `DemoSamplerWrapper` class. We have provided an example of how to use this wrapper along with a demonstration pickle file in the [demo_learning_curriculum](robosuite/scripts/demo_learning_curriculum.py) script.