# RoboTurk Datasets

[RoboTurk](https://roboturk.stanford.edu/) is a crowdsourcing platform developed in order to enabled collecting large-scale manipulation datasets. Below, we describe RoboTurk datasets that are compatible with robosuite.

## Updated Datasets compatible with v1.0

TODO: put this section in after the action-spaces project is complete, and we have numbers for all of this.

Several changes have taken place in robosuite from `v0.3` to `v1.0`. In order to adapt the RoboTurk pilot datasets to our new environments and robot controllers, we attempted to playback the demonstrations by having each robot controller try to track the seqeunce of robot states in each demonstrations, resulting in a new set of demonstrations where the robot tried to reproduce the motions from the original demonstrations. However, this form of playback is not guaranteed to result in successful demonstrations, even though the original demonstrations were all successful. We filtered out the demonstrations that could not be played back successfully, resulting in datasets of reduced size.

We report some dataset metrics below. The datasets have the same structure as the ones collected by the `collect_human_demonstrations.py` script.

## Original Datasets compatible with v0.3

We collected a large-scale dataset on the `SawyerPickPlace` and `SawyerNutAssembly` tasks using the [RoboTurk](https://crowdncloud.ai/) platform. Crowdsourced workers collected these task demonstrations remotely. It consists of **1070** successful `SawyerPickPlace` demonstrations and **1147** successful `SawyerNutAssembly` demonstrations.

We are providing the dataset in the hopes that it will be beneficial to researchers working on imitation learning. Large-scale imitation learning has not been explored much in the community; it will be exciting to see how this data is used.

You can download the dataset [here](http://cvgl.stanford.edu/projects/roboturk/RoboTurkPilot.zip).

After unzipping the dataset, the following subdirectories can be found within the `RoboTurkPilot` directory.

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
