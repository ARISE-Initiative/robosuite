# Benchmarking

Benchmarking results of standard policy learning algorithms.

## v1.0

We provide a standardized set of benchmarking experiments as baselines for future experiments. Specifically, we test [Soft Actor-Critic](https://arxiv.org/abs/1812.05905), a state of the art model-free RL algorithm, on a select combination of tasks (all), robots (Panda, Sawyer), and controllers (OSC_POSE, JOINT_VELOCITY). Our experiments were implemented and executed in an extended version of [rlkit](https://github.com/vitchyr/rlkit), a popular PyTorch-based RL framework and algorithm library. For ease of replicability, we have released our official benchmarking results on a [separate repository](https://github.com/ARISE-Initiative/robosuite-v1-benchmarking).

All agents were trained for 1500 epochs with 500 steps per episode, and utilize the same standardized algorithm hyperparameters (see our benchmarking repo above for exact parameter values). We normalize the our per-step rewards to 1.0 such that the maximum possible per-episode return is 500. Below, we show the per-task experiments conducted, with each experiment's training curve showing the evaluation return mean's average and standard deviation over three random seeds.

### Block Lifting
For the Block Lifting task, both OSC_POSE and JOINT_VELOCITY were tested on Panda and Sawyer robots.

![sac_lift](../images/benchmarking/sac_lift.png)

### Block Stacking
For the Block Stacking task, both OSC_POSE and JOINT_VELOCITY were tested on Panda and Sawyer robots.

![sac_stack](../images/benchmarking/sac_stack.png)

### Pick-and-Place
For the Pick-and-Place task, single-object simplified variations of the task were used. Specifically, PickPlaceCan and PickPlaceMilk were both tested on Panda and Sawyer robots with OSC_POSE.

PickPlaceCan
![sac_pick_place_can](../images/benchmarking/sac_pick_place_can.png)

PickPlaceMilk
![sac_pick_place_milk](../images/benchmarking/sac_pick_place_milk.png)

### Nut Assembly
For the Nut Assembly task, a single-object simplified variation of the task were used. Specifically, NutAssemblyRound was tested on Panda and Sawyer robots with OSC_POSE.

![sac_nut_assembly_round](../images/benchmarking/sac_nut_assembly_round.png)

### Door Opening
For the Door Opening task, both OSC_POSE and JOINT_VELOCITY were tested on Panda and Sawyer robots.

![sac_door](../images/benchmarking/sac_door.png)

### Table Wiping
For the Table Wiping task, both OSC_POSE and JOINT_VELOCITY were tested on Panda and Sawyer robots.

![sac_wipe](../images/benchmarking/sac_wipe.png)

### Two Arm Lift
For the Two Arm Lift task, OSC_POSE was tested on a Panda-Panda pair and Sawyer-Sawyer pair of robots.

![sac_two_arm_lift](../images/benchmarking/sac_two_arm_lift.png)

### Two Arm Peg in Hole
For the Two Arm Peg in Hole task, OSC_POSE was tested on a Panda-Sawyer pair.

![sac_two_arm_peg_in_hole](../images/benchmarking/sac_two_arm_peg_in_hole.png)

### Two Arm Handover
For the Two Arm Handover task, OSC_POSE was tested on a Panda-Panda pair and Sawyer-Sawyer pair of robots.

![sac_two_arm_handover](../images/benchmarking/sac_two_arm_handover.png)



## v0.3

- Please see the [Surreal](http://svl.stanford.edu/assets/papers/fan2018corl.pdf) paper for benchmarking results. Code to reproduce results available [here](https://github.com/SurrealAI/surreal).
- For imitation learning results on [RoboTurk](https://roboturk.stanford.edu/) datasets please see the original [RoboTurk](https://arxiv.org/abs/1811.02790) paper and also the [IRIS](https://arxiv.org/abs/1911.05321) paper.