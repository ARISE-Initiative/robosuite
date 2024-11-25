---
layout: main
title: robosuite
subtitle: 
project_tagline: "A Modular Simulation Framework and Benchmark for Robot Learning"
description: "robosuite: A Modular Simulation Framework and Benchmark for Robot Learning"
videoId: "robosuite-v1.5-video.mp4"
---

![pull figure]({{ 'assets/images/gallery_logo.jpg' | absolute_url }})

**robosuite** is a simulation framework powered by the [MuJoCo](http://mujoco.org/) physics engine for robot learning. It also offers a suite of benchmark environments for reproducible research.
The current release (v1.5) features diverse robot embodiments (including humanoids), custom robot composition, composite controllers (including whole body controllers), more teleoperation devices and photo-realistic rendering. 
This project is part of the broader [Advancing Robot Intelligence through Simulated Environments (ARISE) Initiative](https://github.com/ARISE-Initiative), with the aim of lowering the barriers of entry for cutting-edge research at the intersection of AI and Robotics.

# New Releases

- [10/28/2024] **v1.5**: Added support for diverse robot embodiments (including humanoids), custom robot composition, composite controllers (including whole body controllers), more teleoperation devices, photo-realistic rendering. [[release notes]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.5.0) [[documentation]](http://robosuite.ai/docs/overview.html)

- [11/15/2022] **v1.4**: Backend migration to DeepMind's official [MuJoCo Python binding](https://github.com/deepmind/mujoco), robot textures, and bug fixes :robot: [[video spotlight]](https://youtu.be/2IwDSbj9igQ) [[release notes]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.4.0) [[documentation]](http://robosuite.ai/docs/v1.4/)

- [10/19/2021] **v1.3**: Ray tracing and physically based rendering tools :sparkles: and access to additional vision modalities 🎥 [[video spotlight]](https://www.youtube.com/watch?v=2xesly6JrQ8) [[release notes]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.3) [[documentation]](http://robosuite.ai/docs/v1.3/)

- [02/17/2021] **v1.2**: Added observable sensor models and dynamics randomization [[release notes]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.2)

- [12/17/2020] **v1.1**: Refactored infrastructure and standardized model classes for much easier environment prototyping [[release notes]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.1)

# Video Overview

{% include video/rawVideo.html fileName=page.videoId %}

# Founding Team

{% include founding_team.html %}

# Core Members

{% include core_members.html %}

# Citation

```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n and Abhishek Joshi and Kevin Lin and Soroush Nasiriany and Yifeng Zhu},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```
