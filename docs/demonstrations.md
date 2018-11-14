# Demonstrations

## RoboTurk Dataset

We collected a large-scale dataset on the `SawyerPickPlace` and `SawyerNutAssembly` tasks using the [RoboTurk](https://crowdncloud.ai/) platform. Crowdsourced workers collected these task demonstrations remotely. It consists of **1070** successful `SawyerPickPlace` demonstrations and **1147** successful `SawyerNutAssembly` demonstrations.

We are providing the dataset in the hopes that it will be beneficial to researchers working on imitation learning. Large-scale imitation learning has not been explored much in the community; it will be exciting to see how this data is used.  

You can download the dataset [here](http://cvgl.stanford.edu/projects/roboturk/RoboTurkPilot.zip).

After unzipping the dataset, the following subdirectories can be found within the `RoboTurkPilot` directory. Every directory has the same structure as the demonstrations explained above. 

- **bins-full**
  - The set of complete demonstrations on the full `SawyerPickPlace` task. Every demonstration consists of the Sawyer arm placing one of each object into its corresponding bin.
- **bins-Milk**
  - A postprocessed, segmented set of demonstrations that corresponds to the `SawyerPickPlaceMilk` task. Every demonstration consists of the Sawyer arm placing a can into its corresponding bin. 
- **bins-Bread**
  - A postprocessed, segmented set of demonstrations that corresponds to the `SawyerPickPlaceBread` task. Every demonstration consists of the Sawyer arm placing a loaf of bread into its corresponding bin. 
- **bins-Cereal**
  - A postprocessed, segmented set of demonstrations that corresponds to the `SawyerPickPlaceCereal` task. Every demonstration consists of the Sawyer arm placing a cereal box into its corresponding bin. 
- **bins-Can**
  - A postprocessed, segmented set of demonstrations that corresponds to the `SawyerPickPlaceCan` task. Every demonstration consists of the Sawyer arm placing a can into its corresponding bin. 
- **pegs-full**
  - The set of complete demonstrations on the full `SawyerNutAssembly` task. Every demonstration consists of the Sawyer arm fitting a square nut and a round nut onto their corresponding pegs. 
- **pegs-SquareNut**
  - A postprocessed, segmented set of demonstrations that corresponds to the `SawyerNutAssemblySquare` task. Every demonstration consists of the Sawyer arm fitting a square nut onto its corresponding peg. 
- **pegs-RoundNut**
  - A postprocessed, segmented set of demonstrations that corresponds to the `SawyerNutAssemblyRound` task. Every demonstration consists of the Sawyer arm fitting a round nut onto its corresponding peg. 

## Structure of collected demonstrations

Every set of demonstrations is collected as a directory. Every directory contains a `models` subdirectory and a `demo.hdf5` file. The `models` subdirectory contains an xml file per demonstration, where the xml corresponds to the MuJoCo simulation model that was used during that demonstration. 

The `demo.hdf5` file is structured as follows.

- data (group)

  - date (attribute) - date of collection

  - time (attribute) - time of collection

  - repository_version (attribute) - repository version used during collection

  - env (attribute) - environment name on which demos were collected

    

  - demo1 (group) - group for the first demonstration (every demonstration has a group)

    - model_file (attribute) - name of corresponding model xml in `models` directory

    - states (dataset) - flattened mujoco states, ordered by time

    - joint_velocities (dataset) - joint velocities applied during the demonstration

    - gripper_actuations (dataset) - gripper controls applied during demonstration

    - right_dpos (dataset) - end effector delta position command for single arm robot or right arm

    - right_dquat (dataset) - end effector delta rotation command for single arm robot or right arm

    - left_dpos (dataset) - end effector delta position command for left arm (bimanual robot only)

    - left_dquat (dataset) - end effector delta rotation command for left arm (bimanual robot only)

      

  - demo2 (group) - group for the second demonstration

    ... 

    (and so on)


To see a simple example of how to read demonstrations, please see [playback_demonstrations_from_hdf5](https://github.com/StanfordVL/robosuite/blob/master/robosuite/scripts/playback_demonstrations_from_hdf5.py).

## Collecting your own demonstrations

Demonstrations can be collected by either using a keyboard or using a [SpaceNavigator 3D Mouse](https://www.3dconnexion.com/spacemouse_compact/en/) with the [collect_human_demonstrations](https://github.com/StanfordVL/robosuite/blob/master/robosuite/scripts/collect_human_demonstrations.py) script. It takes the following arguments.

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

### SpaceNavigator 3D Mouse controls

|          Control          |                Command                |
| :-----------------------: | :-----------------------------------: |
|       Right button        |           reset simulation            |
|    Left button (hold)     |             close gripper             |
|   Move mouse laterally    |  move arm horizontally in x-y plane   |
|   Move mouse vertically   |          move arm vertically          |
| Twist mouse about an axis | rotate arm about a corresponding axis |
|      ESC (keyboard)       |                 quit                  |




 
