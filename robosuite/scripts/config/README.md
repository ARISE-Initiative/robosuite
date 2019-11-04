Configurations
======

The config directory provides a set of default configuration files that hold parameters relevant to specific tasks or subsystems (e.g.: controller-specific values).
Below we differentiate between the types of config files and provide descriptions of individual parameter values that each file holds:

controller_config
------
Overview: Provides default parameters for all robot arm controllers supported by `arm_controller.py`. Specifically, the following controllers are supported:

    position
    position_orientation
    position_yaw
    joint_impedance
    joint_torque
    joint_velocity
    
Any of the above controllers may be specified for a given environment by passing the controller name via the `controller` arg when calling robosuite's `make()` function.

Below are descriptions of the individual controller parameters:
####general controller settings
* control_freq: Frequency (Hz) of the policy
* use_impedance: Whether to use impedance control
* use_delta_impedance: Whether to use delta impedance as the action
* control_range": What range to scale control input from the policy (will be used as +/- max/min respectively)
* interpolation: What type of interpolation to use for trajectory planning. Can be "linear" or "cubic". Removing this parameter will default to None

####position-orientation controller
* control_range_ori: What range to scale orientation deltas (will be used as +/- max/min respectively)
* control_range_pos: What range to scale position deltas (will be used as +/- max/min respectively)
* initial_impedance_pos: What impedance to use for position (either constant, or initial)
* initial_impedance_ori: What impedance to use for orientation (either constant, or initial)

####joint-velocity controller
* kv: Kv values for each of the joints
* damping_max: Max damping values (may be per joint or per dimension)
* damping_min: Min damping value (may be per joint or per dimension)
* kp_max: Max kp values (may be per joint or per dimension)
* kp_min: Min kp values (may be per joint or per dimension)
* damping_max_abs_delta: Max range of damping delta used as +/- max/min respectively
* initial_damping: What damping to use (either constant, or initial)
* kp_max_abs_delta: Max kp delta value +/- max/min respectively)
* inertia_decoupling: for joint torques, decoupling with inertia matrix

[task_name]_task_config
--------
Overview: Provides default parameters relevant to individual task environments within robosuite.

Below are descriptions of the individual task parameters:
####general observation settings
* use_camera_obs: Use images as observations for the policy
* use_object_obs: Use ground truth object as observations for the policy
* use_prev_act_obs: Use previous action as part of the observation for the policy
* use_contact_obs: Use the force-torque sensor as contact sensor and take observations from it
* obs_stack_size: Size of the observation stack (default 1)
* only_cartesian_obs: Use only cartesian measurements as observations (e.g. ee pos and vel)
* camera_name: Name of the camera to get observations from
* camera_res: Resolution of the images (we assume square)

####task-specific parameters: FreeSpaceTraj
* acc_vp_reward_mult",help="Multiplier for the num of previously crossed points to add to reward at each step
* action_delta_penalty",help="How much to weight the mean of the delta in the action when penalizing the robot
* dist_threshold: Max dist before end effector is considered to be touching something
* distance_penalty_weight: Weight for how much getting far away from a via point contributes to reward
* distance_reward_weight: Weight for how much getting close to a via point contributes to reward
* ee_accel_penalty: How much to weight the acceleration of the end effector when determining reward
* end_bonus_multiplier: Multiplier for bonus for finishing episode early
* energy_penalty: How much to penalize the mean energy used by the joints
* num_already_checked",help="How many of the via points have already been hit (to simplify the task without changing the observation space)
* num_via_points: Number of points the robot must go through
* point_randomization",help="Absolute value of variation in points of square.
* random_point_order: Whether or not to switch randomly between going clockwise and going counter-clockwise
* timestep_penalty: Amount of reward subtracted at each timestep
* use_debug_cube: Whether to use a fixed 8 corners of a cube as the via points
* use_debug_point: Whether to use a single fixed point
* use_debug_point_two: Whether to use the second single fixed point
* use_debug_square: Whether to use a fixed 4 corners of a square as the via points
* obstacle_avoidance: Whether to use obstacle avoidance version
* use_delta_distance_reward: Whether to only reward agent for getting closer to the point
* via_point_reward: Amount of reward added for reaching a via point

####task-specific parameters: Wiping
* arm_collision_penalty: Penalty in the reward for colliding with the arm
* cnn_small",help="When using CNN, if we want it small
* distance_multiplier: Multiplier of the dense reward to the mean distance to pegs
* distance_th_multiplier: Multiplier inside the tanh for the mean distance to pegs
* draw_line: Limit the desired position of the ee
* excess_force_penalty_mul: Multiplier for the excess of force applied to compute the penalty
* line_width",help="Width of the painted line
* n_units_x: Number of units to divide the table in x
* n_units_y: Number of units to divide the table in y
* num_sensors: Probability of place a sensor
* pressure_threshold_max: Max force the robot can apply on the environment
* prob_sensor: Probability of place a sensor
* shear_threshold: Shear force threshold to deactivate a sensor
* table_friction: Friction of the table
* table_friction_std: Std for the friction of the table
* table_height: Height of the table
* table_height_std: Standard dev of the height of the table
* table_rot_x: Std dev of the rotation of the table around x
* table_rot_y: Std dev of the rotation of the table around y
* table_size: Size of the table (assumed square surface)
* touch_threshold: Pressure threshold to deactivate a sensor
* two_clusters",help="Creates two clusters of units to wipe
* unit_wiped_reward: Reward for wiping one unit (sensor or peg)
* wipe_contact_reward: Reward for maintaining contact
* with_pos_limits: Limit the desired position of the ee
* with_qinits",help="Picks among a set of initial qs

####task-specific parameters: Door 
* change_door_friction: Resets door damping and friction to a uniformly distributed random value specified by the ranges below
* door_damping_max: Maximum door damping
* door_damping_min: Minimum door damping
* door_friction_max: Maximum door friction
* door_friction_min: Minimum door friction
* gripper_on_handle: Initialize robot position within close proximity to door handle
* handle_reward: Reward for touching handle
* use_door_state: Use door hinge angle and handle pos as obs

[alg_name]_alg_config TODO
--------