# Demos

We provide a collection of [demo scripts](https://github.com/Unknown-Initiative/robosuite-dev/tree/v1.0/robosuite/demos) to showcase the functionalities in **robosuite**.

### Modular Simulation Creation
The `demo_random_action.py` sciprt is the starter demo script that you should try first. It highlights the modular design of our simulated environments. It enables users to create new simulation instances by choosing one [environment](modules/environments), one or more [robots](modules/robots), and their [controllers](modules/controllers) from the command line. The script creates an environment instance and controls the robots with uniform random actions drawn from the controller-specific action space. The list of all environments, robots, controllers, and gripper types supported in the current version of **robosuite** are defined by `suite.ALL_ENVIRONMENTS`, `suite.ALL_ROBOTS`, `suite.ALL_CONTROLLERS`, and `suite.ALL_GRIPPERS` respectively.


### Controller Test
The `demo_control.py` script demonstrates the various functionalities of each controller available within **robosuite**.
For a given controller, runs through each dimension and executes a perturbation `test_value` from its
neutral (stationary) value for a certain amount of time "steps_per_action", and then returns to all neutral values
for time `steps_per_rest` before proceeding with the next action dim.
For example, given that the expected action space of the `OSC_POSE` controller (without a gripper) is `(dx, dy, dz, droll, dpitch, dyaw)`, the testing sequence of actions over time will be:

```
***START OF DEMO***
( dx,  0,  0,  0,  0,  0, grip)     <-- Translation in x-direction      for 'steps_per_action' steps
(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
(  0, dy,  0,  0,  0,  0, grip)     <-- Translation in y-direction      for 'steps_per_action' steps
(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
(  0,  0, dz,  0,  0,  0, grip)     <-- Translation in z-direction      for 'steps_per_action' steps
(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
(  0,  0,  0, dr,  0,  0, grip)     <-- Rotation in roll (x) axis       for 'steps_per_action' steps
(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
(  0,  0,  0,  0, dp,  0, grip)     <-- Rotation in pitch (y) axis      for 'steps_per_action' steps
(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
(  0,  0,  0,  0,  0, dy, grip)     <-- Rotation in yaw (z) axis        for 'steps_per_action' steps
(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
***END OF DEMO***
```

Thus the `OSC_POSE` controller should be expected to sequentially move linearly in the x direction first, then the y direction, then the z direction, and then begin sequentially rotating about its x-axis, then y-axis, then z-axis. Please reference the documentation of [Controllers](modules/controllers) for an overview of each controller. Controllers are expected to behave in a generally controlled manner, according to their control space. The expected sequential qualitative behavior during the test is described below for each controller:

* `OSC_POSE`: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis, z-axis, relative to the global coordinate frame
* `OSC_POSITION`: Gripper moves sequentially and linearly in x, y, z direction, relative to the global coordinate frame
* `IK_POSE`: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis, z-axis, relative to the local robot end effector frame
* `JOINT_POSITION`: Robot Joints move sequentially in a controlled fashion
* `JOINT_VELOCITY`: Robot Joints move sequentially in a controlled fashion
* `JOINT_TORQUE`: Unlike other controllers, joint torque controller is expected to act rather lethargic, as the "controller" is really just a wrapper for direct torque control of the mujoco actuators. Therefore, a "neutral" value of 0 torque will not guarantee a stable robot when it has non-zero velocity!

### Gripper Selection
The `demo_gripper_selection.py` script shows you how to select gripper for an environment. This is controlled by `gripper_type` keyword argument. The set of all grippers is defined by the global variable `robosuite.ALL_GRIPPERS`.

![collection of grippers](images/gripper_collection.png)

### demo_collect_and_playback_data.py

### demo_gym_functionality.py


### demo_learning_curriculum.py


### demo_device_control.py


### demo_pygame_renderer.py



### demo_domain_randomization.py


### demo_gripper_interaction.py



### demo_video_recording.py