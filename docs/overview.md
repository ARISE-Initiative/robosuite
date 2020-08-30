# Overview

![gallery of_environments](images/gallery.png)

**robosuite** is a robotic simulation library powered by the [MuJoCo physics engine](http://mujoco.org/) for reproducible robotics research. The current release focuses on reinforcement and imitation learning for robot manipulation. 

Reinforcement learning has been a powerful and generic tool in robotics. Reinforcement learning combined with deep neural networks, i.e., *deep reinforcement learning* (DRL), has achieved some exciting successes in a variety of robot control problems. However, the challenges of reproducibility and replicability in DRL and robotics have impaired research progress. Our goal is to provide an accessible set of benchmarking tasks that facilitates a fair and rigorus evaluation and improves our understanding of new methods.

This framework was originally developed since late 2017 by researchers in [Stanford Vision and Learning Lab](http://svl.stanford.edu/) (SVL) as an internal tool for robot learning research. Now it is actively maintained and used for robotics research projects in SVL and the Robot Perception and Learning (RPL) Lab at UT-Austin.

This release of **robosuite** contains a set of benchmarking manipulation tasks and a modularized design of APIs for building new environments. We highlight these primary features below:

* [**standardized tasks**](modules/environments): a set of standardized manipulation tasks of large diversity and varying complexity and RL benchmarking results for reproducible research;
* [**procedural generation**](modules/overview): modular APIs for programmatically creating new scenes and new tasks as a combinations of robot models, arenas, and parameterized 3D objects;
* [**controller supports**](modules/controllers): a selection of controller types to command the robots, such as joint-space velocity control, inverse kinematics control, operational space control, and 3D motion devices for teleoperation;
* [**multi-modal sensors**](modules/sensors): heterogeneous types of sensory signals, including low-level physical states, RGB cameras, depth maps, and proprioception;
* [**human demonstrations**](algorithms/demonstrations): utilities for collecting human demonstrations, replaying demonstration datasets, and leveraging demonstration data for learning.


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
