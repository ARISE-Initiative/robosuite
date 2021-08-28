from robosuite.robots.robot import Robot


class Manipulator(Robot):
    """
    Initializes a manipulator robot simulation object, as defined by a single corresponding robot arm XML and
    associated gripper XML
    """

    def _load_controller(self):
        raise NotImplementedError

    def control(self, action, policy_step=False):
        raise NotImplementedError

    def grip_action(self, gripper, gripper_action):
        """
        Executes @gripper_action for specified @gripper

        Args:
            gripper (GripperModel): Gripper to execute action for
            gripper_action (float): Value between [-1,1] to send to gripper
        """
        actuator_idxs = [
            self.sim.model.actuator_name2id(actuator) for actuator in gripper.actuators
        ]
        gripper_action_actual = gripper.format_action(gripper_action)
        # rescale normalized gripper action to control ranges
        ctrl_range = self.sim.model.actuator_ctrlrange[actuator_idxs]
        bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        applied_gripper_action = bias + weight * gripper_action_actual
        self.sim.data.ctrl[actuator_idxs] = applied_gripper_action

    def visualize(self, vis_settings):
        """
        Do any necessary visualization for this manipulator

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "robots" and "grippers" keyword as well as any other
                robot-specific options specified.
        """
        super().visualize(vis_settings=vis_settings)
        self._visualize_grippers(visible=vis_settings["grippers"])

    def _visualize_grippers(self, visible):
        """
        Visualizes the gripper site(s) if applicable.

        Args:
            visible (bool): True if visualizing grippers, else False
        """
        raise NotImplementedError

    @property
    def action_limits(self):
        raise NotImplementedError

    @property
    def dof(self):
        """
        Returns:
            int: degrees of freedom of the robot (with grippers).
        """
        # Get the dof of the base robot model
        dof = super().dof
        for gripper in self.robot_model.grippers.values():
            dof += gripper.dof
        return dof

    @property
    def ee_ft_integral(self):
        """
        Returns:
            float or dict: either single value or arm-specific entries specifying the integral over time of the applied
                ee force-torque for that arm
        """
        raise NotImplementedError

    @property
    def ee_force(self):
        """
        Returns:
            np.array or dict: either single value or arm-specific entries specifying the force applied at the force sensor
                at the robot arm's eef
        """
        raise NotImplementedError

    @property
    def ee_torque(self):
        """
        Returns:
            np.array or dict: either single value or arm-specific entries specifying the torque applied at the torque
                sensor at the robot arm's eef
        """
        raise NotImplementedError

    @property
    def _hand_pose(self):
        """
        Returns:
            np.array or dict: either single value or arm-specific entries specifying the eef pose in base frame of
                robot.
        """
        raise NotImplementedError

    @property
    def _hand_quat(self):
        """
        Returns:
            np.array or dict: either single value or arm-specific entries specifying the eef quaternion in base frame
                of robot.
        """
        raise NotImplementedError

    @property
    def _hand_total_velocity(self):
        """
        Returns:
            np.array or dict: either single value or arm-specific entries specifying the total eef velocity
                (linear + angular) in the base frame as a numpy array of shape (6,)
        """
        raise NotImplementedError

    @property
    def _hand_pos(self):
        """
        Returns:
            np.array or dict: either single value or arm-specific entries specifying the position of eef in base frame
                of robot.
        """
        raise NotImplementedError

    @property
    def _hand_orn(self):
        """
        Returns:
            np.array or dict: either single value or arm-specific entries specifying the orientation of eef in base
                frame of robot as a rotation matrix.
        """
        raise NotImplementedError

    @property
    def _hand_vel(self):
        """
        Returns:
            np.array or dict: either single value or arm-specific entries specifying the velocity of eef in base frame
                of robot.
        """
        raise NotImplementedError

    @property
    def _hand_ang_vel(self):
        """
        Returns:
            np.array or dict: either single value or arm-specific entries specifying the angular velocity of eef in
                base frame of robot.
        """
        raise NotImplementedError
