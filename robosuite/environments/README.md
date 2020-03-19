Environments
============
#TODO: UPDATE!!

Environments load scene models from [Models](../models) and create a simulation of the task. In addition to model instantiation, the environment classes implement the definitions of observations, reward functions, success conditions, etc. that are required APIs for reinforcement learning.

Currently, we have implemented six manipulation tasks with two robots, _Sawyer_ and _Baxter_. Our rationale of designing these tasks is to offer single-arm and bimanual manipulation tasks of large diversity and varying complexity. In the default settings, The robots are controlled via joint velocity. There are three types of observations: proprioceptive feature (`robot-state`), object-centric feature (`object-state`) and RGB/RGB-D image (`image`). Proprioceptive observations contain: `cos` and `sin` of robot joint positions, robot joint velocities and current configuration of the gripper. Object-centric observations contain task-specific features. Image observations are RGB/RGB-D images (`256 x 256` by default). When trained on states, the agent receives `robot-state` and `object-state` observations. When trained on pixels, the agent receives `robot-state` and `image` observations.

All environments should inherit a base [MujocoEnv](base.py) class which registers the environment. The environments for each robot type should inherit their corresponding base environment classes, such as [SawyerEnv](sawyer.py) and [BaxterEnv](baxter.py), which define some robot-specific logic. All environments are automatically added to the registry and can be instantiated by the `make` function.

```python
import robosuite as suite

# below provide examples of initializing environments for different purposes
# see all possible configurations in environment python classes

# create an environment for screen visualization
env = suite.make(
    "SawyerLift",      # environment name (see below)
    has_renderer=True, # create on-screen renderer
)

# create an environment for learning on states
env = suite.make(
    "SawyerLift",
    has_renderer=False,           # no on-screen renderer
    has_offscreen_renderer=False, # no off-screen renderer
    use_object_obs=True,          # use object-centric feature
    use_camera_obs=False,         # no camera observations
)

# create an environment for learning on pixels
env = suite.make(
    "SawyerLift",
    has_renderer=False,          # no on-screen renderer
    has_offscreen_renderer=True, # off-screen renderer is required for camera observations
    ignore_done=True,            # (optional) never terminates episode
    use_camera_obs=True,         # use camera observations
    camera_height=84,            # set camera height
    camera_width=84,             # set camera width
    camera_name='agentview',     # use "agentview" camera
    use_object_obs=False,        # no object feature when training on pixels
    reward_shaping=True          # (optional) using a shaping reward
)
```

Task Descriptions
-----------------

We provide a brief description of each environment below:

[SawyerLift](sawyer_lift.py): A cube is placed on the tabletop. The Sawyer robot is rewarded for lifting the cube with a parallel-jaw gripper. We randomize the size and the placement of the cube.

[SawyerStack](sawyer_stack.py): A red cube and a green cube are placed on the tabletop. The Sawyer robot is rewarded for lifting the red cube with a parallel-jaw gripper and stack it on top of the green cube.

[BaxterPegInHole](baxter_peg_in_hole.py): The Baxter robot holds a board with a squared hole in the center in its right hand, and a long stick in the left hand. The goal is to move both arms to insert the peg into the hole.

[BaxterLift](baxter_lift.py): A pot with two handles is placed on the tabletop. The Baxter robot is rewarded for lifting the pot above the table by a threshold while not tilting the pot over 30 degrees. Thus the robot has to coordinate its two hands to grasp the handles and balance the pot.

[SawyerPickPlace](sawyer_pick_place.py): The Sawyer robot tackles a pick-and-place task, where the goal is to pick four objects from each category in a bin and to place them into their corresponding containers. This task also include several variants of easier mode, which only consists of one object.

[SawyerNutAssembly](sawyer_nut_assembly.py): Two colored pegs are mounted to the tabletop. The Sawyer robot needs to declutter four nuts lying on top of each other and assembles them onto their corresponding pegs. This task also include several variants of easier mode, which only consists of one nut.
