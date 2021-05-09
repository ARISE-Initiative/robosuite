# RoboTurk Datasets

[RoboTurk](https://roboturk.stanford.edu/) is a crowdsourcing platform developed in order to enabled collecting large-scale manipulation datasets. Below, we describe RoboTurk datasets that are compatible with robosuite.

## Updated Datasets compatible with v1.0+

We are currently in the process of organizing a standardized dataset for our benchmarking tasks, which will be made available soon and compatible with v1.2.0+. In the meantime, we have provided a [small-scale dataset](https://drive.google.com/drive/folders/1LLkuFnRdqQ6xn1cYzkbJUs_DreaAvN7i?usp=sharing) of expert demonstrations on two of our tasks.

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
