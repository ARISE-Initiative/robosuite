from typing import Dict, List
import numpy as np

from robosuite.controllers.controller import Controller
from robosuite.utils.control_utils import *

# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp"}


class JointPositionController(Controller):
    """
    Controller for controlling robot arm via impedance control. Allows position control of the robot's joints.

    NOTE: Control input actions assumed to be taken relative to the current joint positions. A given action to this
    controller is assumed to be of the form: (dpos_j0, dpos_j1, ... , dpos_jn-1) for an n-joint robot

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or Iterable of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or Iterable of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or Iterable of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or Iterable of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        kp (float or Iterable of float): positional gain for determining desired torques based upon the joint pos error.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        damping_ratio (float or Iterable of float): used in conjunction with kp to determine the velocity gain for
            determining desired torques based upon the joint pos errors. Can be either be a scalar (same value for all
            action dims), or a list (specific values for each dim)

        impedance_mode (str): Impedance mode with which to run this controller. Options are {"fixed", "variable",
            "variable_kp"}. If "fixed", the controller will have fixed kp and damping_ratio values as specified by the
            @kp and @damping_ratio arguments. If "variable", both kp and damping_ratio will now be part of the
            controller action space, resulting in a total action space of num_joints * 3. If "variable_kp", only kp
            will become variable, with damping_ratio fixed at 1 (critically damped). The resulting action space will
            then be num_joints * 2.

        kp_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is set to either
            "variable" or "variable_kp". This sets the corresponding min / max ranges of the controller action space
            for the varying kp values. Can be either be a 2-list (same min / max for all kp action dims), or a 2-list
            of list (specific min / max for each kp dim)

        damping_ratio_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is
            set to "variable". This sets the corresponding min / max ranges of the controller action space for the
            varying damping_ratio values. Can be either be a 2-list (same min / max for all damping_ratio action dims),
            or a 2-list of list (specific min / max for each damping_ratio dim)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        qpos_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the magnitude
            of a calculated goal joint position will be clipped. Can be either be a 2-list (same min/max value for all
            joint dims), or a 2-list of list (specific min/max values for each dim)

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current joint position to
            the goal joint position during each timestep between inputted actions

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(
        self,
        sim,
        joint_indexes,
        actuator_range,
        ref_name=None,
        input_max=1,
        input_min=-1,
        output_max=0.05,
        output_min=-0.05,
        kp=50,
        damping_ratio=1,
        impedance_mode="fixed",
        kp_limits=(0, 300),
        damping_ratio_limits=(0, 100),
        policy_freq=20,
        lite_physics=False,
        qpos_limits=None,
        interpolator=None,
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):

        super().__init__(
            sim,
            ref_name=ref_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            part_name=kwargs.get("part_name", None),
            naming_prefix=kwargs.get("naming_prefix", None),
            lite_physics=lite_physics,
        )

        self.joint_indexes = joint_indexes

        # Control dimension
        self.control_dim = len(joint_indexes["joints"])

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # limits
        self.position_limits = np.array(qpos_limits) if qpos_limits is not None else qpos_limits

        # kp kd
        self.kp = self.nums2array(kp, self.control_dim)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        print(f'self.kp: {self.kp}')
        print(f'self.kd: {self.kd}')
        # import ipdb; ipdb.set_trace()

        # # print out joint angles of all joints
        # for joint in self.robots.joints:
        #     print(f"joint: {joint} {self.sim.data.get_joint_qpos(joint)}")

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], self.control_dim)
        self.kp_max = self.nums2array(kp_limits[1], self.control_dim)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], self.control_dim)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], self.control_dim)

        # Verify the proposed impedance mode is supported
        assert impedance_mode in IMPEDANCE_MODES, (
            "Error: Tried to instantiate OSC controller for unsupported "
            "impedance mode! Inputted impedance mode: {}, Supported modes: {}".format(impedance_mode, IMPEDANCE_MODES)
        )

        # Impedance mode
        self.impedance_mode = impedance_mode

        # Add to control dim based on impedance_mode
        if self.impedance_mode == "variable":
            self.control_dim *= 3
        elif self.impedance_mode == "variable_kp":
            self.control_dim *= 2

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize
        self.goal_qpos = None

    def update_base_pose(self):
        pass

    def set_goal(self, action, set_qpos=None):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_qpos (Iterable): If set, overrides @action and sets the desired absolute joint position goal state

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        # Update state
        self.update()

        # Parse action based on the impedance mode, and update kp / kd as necessary
        jnt_dim = len(self.qpos_index)
        if self.impedance_mode == "variable":
            damping_ratio, kp, delta = action[:jnt_dim], action[jnt_dim : 2 * jnt_dim], action[2 * jnt_dim :]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp) * np.clip(damping_ratio, self.damping_ratio_min, self.damping_ratio_max)
        elif self.impedance_mode == "variable_kp":
            kp, delta = action[:jnt_dim], action[jnt_dim:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp)  # critically damped
        else:  # This is case "fixed"
            delta = action

        # Check to make sure delta is size self.joint_dim
        assert len(delta) == jnt_dim, "Delta qpos must be equal to the robot's joint dimension space!"

        if delta is not None:
            scaled_delta = self.scale_action(delta)
        else:
            scaled_delta = None

        self.goal_qpos = set_goal_position(
            scaled_delta, self.joint_pos, position_limit=self.position_limits, set_pos=set_qpos
        )

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Make sure goal has been set
        if self.goal_qpos is None:
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        self.update()

        desired_qpos = None

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            # Linear case
            if self.interpolator.order == 1:
                desired_qpos = self.interpolator.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            desired_qpos = np.array(self.goal_qpos)

        from robosuite.scripts.diffik_nullspace import RobotController

        model = self.sim.model._model
        data = self.sim.data._data
        joint_names = [model.joint(i).name for i in range(model.njnt) if model.joint(i).type != 0]  # Exclude fixed joints
        body_names = [model.body(i).name for i in range(model.nbody) if model.body(i).name not in {"world", "base", "target"}]

        def get_Kn(joint_names: List[str], weight_dict: Dict[str, float]) -> np.ndarray:
            return np.array([weight_dict.get(joint, 1.0) for joint in joint_names])

        nullspace_joint_weights = {
            "robot0_torso_waist_yaw": 100.0,
            "robot0_torso_waist_pitch": 100.0,
            "robot0_torso_waist_roll": 500.0,
            "robot0_l_shoulder_pitch": 4.0,
            "robot0_r_shoulder_pitch": 4.0,
            "robot0_l_shoulder_roll": 3.0,
            "robot0_r_shoulder_roll": 3.0,
            "robot0_l_shoulder_yaw": 2.0,
            "robot0_r_shoulder_yaw": 2.0,
        }
        Kn = get_Kn(joint_names, nullspace_joint_weights)
        end_effector_sites = [ "gripper0_left_grip_site", "gripper0_right_grip_site"]
        robot_config =  {
            'end_effector_sites': end_effector_sites,
            'body_names': body_names,
            'joint_names': joint_names,
            'mocap_bodies': [],
            'initial_keyframe': 'home',
            'nullspace_gains': Kn
        }
        robot = RobotController(model, data, robot_config, input_type="mocap", debug=True)
        # desired_qpos[:] = 0 # + np.random.normal(0, 0.1, len(desired_qpos))

        target_pos = np.array([[-0.419,  0.28 ,  1.11 ],
                                [-0.419, -0.279,  1.11 ]])
        target_ori = np.array([[-0.465, -0.46 ,  0.54 , -0.53 ],
                                [ 0.548, -0.535,  0.487,  0.419]])

        integration_dt = 0.1
        damping = 5e-2
        dt = 0.002        
        Kpos = 0.95
        Kori = 0.95

        max_actuation_val = 100
        torques = robot.solve_ik(
            target_pos=target_pos, 
            target_ori=target_ori, 
            damping=damping,
            integration_dt=integration_dt, 
            max_actuation_val=max_actuation_val,
            Kpos=Kpos, 
            Kori=Kori, 
            update_sim=False)
        self.torques = torques
        # torques = pos_err * kp + vel_err * kd
        # position_error = desired_qpos - self.joint_pos
        # vel_pos_error = -self.joint_vel
        # desired_torque = np.multiply(np.array(position_error), np.array(self.kp)) + np.multiply(vel_pos_error, self.kd)
        # # Return desired torques plus gravity compensations
        # self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation

        # print(f"max joint position error: {np.max(np.abs(position_error))}")

        # Always run superclass call for any cleanups at the end
        super().run_controller()
        return self.torques

    def reset_goal(self):
        """
        Resets joint position goal to be current position
        """
        self.goal_qpos = self.joint_pos

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        if self.impedance_mode == "variable":
            low = np.concatenate([self.damping_ratio_min, self.kp_min, self.input_min])
            high = np.concatenate([self.damping_ratio_max, self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        else:  # This is case "fixed"
            low, high = self.input_min, self.input_max
        return low, high

    @property
    def name(self):
        return "JOINT_POSITION"
