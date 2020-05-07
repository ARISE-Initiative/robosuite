"""Demo of learning curriculum utilities.

Several prior works have demonstrated the effectiveness of altering the
start state distribution of training episodes for learning RL policies.
We provide a generic utility for setting various types of learning 
curriculums which dictate how to sample from demonstration episodes
when doing an environment reset. For more information see the 
`DemoSamplerWrapper` class. 

Related work:

[1] Reinforcement and Imitation Learning for Diverse Visuomotor Skills
Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi,Saran Tunyasuvunakool,
János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
RSS 2018

[2] Backplay: "Man muss immer umkehren"
Cinjon Resnick, Roberta Raileanu, Sanyam Kapoor, Alex Peysakhovich, Kyunghyun Cho, Joan Bruna
arXiv:1807.06919

[3] DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills
Xue Bin Peng, Pieter Abbeel, Sergey Levine, Michiel van de Panne
Transactions on Graphics 2018

[4] Approximately optimal approximate reinforcement learning
Sham Kakade and John Langford
ICML 2002
"""

import os

import robosuite
from robosuite import make
from robosuite.wrappers import DemoSamplerWrapper

# TODO: Demonstrations path is now depreciated. Need to update and/or get new demonstrations!!

if __name__ == "__main__":

    env = make(
        "PickPlace",
        robots="Sawyer",
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        gripper_visualizations=True,
    )

    env = DemoSamplerWrapper(
        env,
        demo_path=os.path.join(
            robosuite.models.assets_root, "demonstrations/SawyerPickPlace"
        ),
        need_xml=True,
        num_traj=-1,
        sampling_schemes=["uniform", "random"],
        scheme_ratios=[0.9, 0.1],
    )

    for _ in range(100):
        env.reset()
        env.viewer.set_camera(0)
        env.render()
        for i in range(100):
            if i == 0:
                reward = env.reward()
                print("reward", reward)
            env.render()
