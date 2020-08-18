# Human Demonstrations

## Collecting Human Demonstrations

We provide teleoperation utilities that allow users to control the robots with input devices, such as the keyboard and the [SpaceMouse](https://www.3dconnexion.com/spacemouse_compact/en/). Such functionality allows us to collect a dataset of human demonstrations for learning. We provide an example script to illustrate how to collect demonstrations. Our [collect_human_demonstrations](robosuite/scripts/collect_human_demonstrations.py) script takes the following arguments:

- `directory:` path to a folder for where to store the pickle file of collected demonstrations
- `environment:` name of the environment you would like to collect the demonstrations for
- `device:` either "keyboard" or "spacemouse"

Our twin project [RoboTurk](http://roboturk.stanford.edu) has collected pilot datasets of more than a thousand demonstrations for two tasks in our Suite via crowdsourcing. You can find detailed information about the [RoboTurk datasets](docs/demonstrations.md#roboturk-dataset) and [demonstration collection](docs/demonstrations.md#collecting-your-own-demonstrations) here.

## Replaying Human Demonstrations

We have included an example script that illustrates how demonstrations can be loaded and played back. Our [playback_demonstrations_from_hdf5](robosuite/scripts/playback_demonstrations_from_hdf5.py) script selects demonstration episodes at random from a demonstration pickle file and replays them. We have included some sample demonstrations for each task at `models/assets/demonstrations`.

## Using Demonstrations for Learning

[Several](https://arxiv.org/abs/1802.09564) [prior](https://arxiv.org/abs/1807.06919) [works](https://arxiv.org/abs/1804.02717) have demonstrated the effectiveness of altering the start state distribution of training episodes for learning RL policies. We provide a generic utility for setting various types of learning curriculums which dictate how to sample from demonstration episodes when doing an environment reset. For more information see the `DemoSamplerWrapper` class. We have provided an example of how to use this wrapper along with a demonstration pickle file in the [demo_learning_curriculum](robosuite/scripts/demo_learning_curriculum.py) script.