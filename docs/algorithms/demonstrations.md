# Human Demonstrations

## Collecting Human Demonstrations

We provide teleoperation utilities that allow users to control the robots with input devices, such as the keyboard, [SpaceMouse](https://www.3dconnexion.com/spacemouse_compact/en/), [DualSense](https://www.playstation.com/en-us/accessories/dualsense-wireless-controller/) and mujoco-gui. Such functionality allows us to collect a dataset of human demonstrations for learning. We provide an example script to illustrate how to collect demonstrations. Our [collect_human_demonstrations](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/collect_human_demonstrations.py) script takes the following arguments:

- `directory:` path to a folder for where to store the pickle file of collected demonstrations
- `environment:` name of the environment you would like to collect the demonstrations for
- `device:` either "keyboard" or "spacemouse" or "dualsense" or "mjgui"
- `renderer:` Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)
- `camera:` Pass multiple camera names to enable multiple views. Note that the "mujoco" renderer must be enabled when using multiple views, while "mjviewer" is not supported.

See the [devices page](https://robosuite.ai/docs/modules/devices.html) for details on how to use the devices.

## Replaying Human Demonstrations

We have included an example script that illustrates how demonstrations can be loaded and played back. Our [playback_demonstrations_from_hdf5](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/playback_demonstrations_from_hdf5.py) script selects demonstration episodes at random from a demonstration pickle file and replays them.


## Existing Datasets

We have included some sample demonstrations for each task at `models/assets/demonstrations`.


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

The [robomimic](https://arise-initiative.github.io/robomimic-web/) framework makes it easy to train policies using your own [datasets collected with robosuite](https://arise-initiative.github.io/robomimic-web/docs/introduction/datasets.html#robosuite-hdf5-datasets). The framework also contains many useful examples for how to integrate hdf5 datasets into your own learning pipeline.

The robosuite repository also has some utilities for using the demonstrations to alter the start state distribution of training episodes for learning RL policies - this have proved effective in [several](https://arxiv.org/abs/1802.09564) [prior](https://arxiv.org/abs/1807.06919) [works](https://arxiv.org/abs/1804.02717). For example, we provide a generic utility for setting various types of learning curriculums which dictate how to sample from demonstration episodes when doing an environment reset. For more information see the `DemoSamplerWrapper` class.

## Warnings
We have verified that deterministic action playback works specifically when playing back demonstrations on the *same machine* that the demonstrations were originally collected upon. However, this means that deterministic action playback is NOT guaranteed (in fact, very unlikely) to work across platforms or even across different machines using the same OS.

While action playback trajectories are quite similar even if not completely identical to the original collected state trajectories, they do tend to drift over time, and should not be relied upon to accurately replicate demonstrations. Instead, we recommend directly setting states to reproduce the collected trajectories, as shown in [playback_demonstrations_from_hdf5](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/playback_demonstrations_from_hdf5.py).