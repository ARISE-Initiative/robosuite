# RoboTurk Datasets

[RoboTurk](https://roboturk.stanford.edu/) is a crowdsourcing platform developed in order to enabled collecting large-scale manipulation datasets. Below, we describe RoboTurk datasets that are compatible with robosuite.

## Datasets compatible with v1.2+

We have collected several human demonstration datasets across several tasks implemented in robosuite as part of the [robomimic](https://arise-initiative.github.io/robomimic-web/) framework. For more information on these datasets, including how to download them and start training policies with them, please see [this link](https://arise-initiative.github.io/robomimic-web/docs/introduction/results.html#downloading-released-datasets).

## Datasets compatible with v0.3

We collected a large-scale dataset on the `SawyerPickPlace` and `SawyerNutAssembly` tasks using the [RoboTurk](https://crowdncloud.ai/) platform. Crowdsourced workers collected these task demonstrations remotely. It consists of **1070** successful `SawyerPickPlace` demonstrations and **1147** successful `SawyerNutAssembly` demonstrations.

We are providing the dataset in the hopes that it will be beneficial to researchers working on imitation learning. Large-scale imitation learning has not been explored much in the community; it will be exciting to see how this data is used.

You can download the dataset [here](http://cvgl.stanford.edu/projects/roboturk/RoboTurkPilot.zip).

**Note:** to get started with this data, we highly recommend using the [robomimic](https://arise-initiative.github.io/robomimic-web/) framework - see [this link](https://arise-initiative.github.io/robomimic-web/docs/introduction/datasets.html#roboturk-pilot-datasets) for more information.

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
