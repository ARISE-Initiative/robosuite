"""
Convenience script to tune a robot's joint positions in a mujoco environment.
Allows keyboard presses to move specific robot joints around in the viewer, and
then prints the current joint parameters upon an inputted command

RELEVANT KEY PRESSES:
    '1 - n' : Sets the active robot joint being tuned to this number. Maximum
        is n which is the number of robot joints
    't' : Toggle between robot arms being tuned (only applicable for multi-arm environments)
    'r' : Resets the active joint values to 0
    'UP_ARROW' : Increment the active robot joint position
    'DOWN_ARROW' : Decrement the active robot joint position
    'RIGHT_ARROW' : Increment the delta joint position change per keypress
    'LEFT_ARROW' : Decrement the delta joint position change per keypress

"""

import argparse

import numpy as np
from pynput.keyboard import Controller, Key, Listener

import robosuite
from robosuite.robots import SingleArm


class KeyboardHandler:
    def __init__(self, env, delta=0.05):
        """
        Store internal state here.

        Args:
            env (MujocoEnv): Environment to use
            delta (float): initial joint tuning increment
        """
        self.env = env
        self.delta = delta
        self.num_robots = len(env.robots)
        self.active_robot_num = 0
        self.active_arm_joint = 1
        self.active_arm = "right"  # only relevant for bimanual robots
        self.current_joints_pos = env.sim.data.qpos[self.active_robot._ref_joint_pos_indexes[: self.num_joints]]

        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    def on_press(self, key):
        """
        Key handler for key presses.

        Args:
            key (int): keycode corresponding to the key that was pressed
        """

        try:
            if key == Key.up:
                # Increment the active joint
                self._update_joint_position(self.active_arm_joint, self.delta)
            elif key == Key.down:
                # Decrement the active joint
                self._update_joint_position(self.active_arm_joint, -self.delta)
            elif key == Key.right:
                # Increment the delta value
                self.delta = min(1.0, self.delta + 0.005)
                # Print out new value to user
                print("Delta now = {:.3f}".format(self.delta))
            elif key == Key.left:
                # Decrement the delta value
                self.delta = max(0, self.delta - 0.005)
                print("Delta now = {:.3f}".format(self.delta))
            # controls for setting active arm
            elif key.char == "0":
                # Notify use that joint indexes are 1-indexed
                print("Joint Indexes are 1-Indexed. Available joints are 1 - {}".format(self.num_joints))
            elif key.char == "1":
                # Make sure range is valid; if so, update this specific joint
                if self._check_valid_joint(1):
                    self.active_arm_joint = 1
                    # Print out to user
                    print("New joint being tuned: {}".format(self.active_arm_joint))
            elif key.char == "2":
                # Make sure range is valid; if so, update this specific joint
                if self._check_valid_joint(2):
                    self.active_arm_joint = 2
                    # Print out to user
                    print("New joint being tuned: {}".format(self.active_arm_joint))
            elif key.char == "3":
                # Make sure range is valid; if so, update this specific joint
                if self._check_valid_joint(3):
                    self.active_arm_joint = 3
                    # Print out to user
                    print("New joint being tuned: {}".format(self.active_arm_joint))
            elif key.char == "4":
                # Make sure range is valid; if so, update this specific joint
                if self._check_valid_joint(4):
                    self.active_arm_joint = 4
                    # Print out to user
                    print("New joint being tuned: {}".format(self.active_arm_joint))
            elif key.char == "5":
                # Make sure range is valid; if so, update this specific joint
                if self._check_valid_joint(5):
                    self.active_arm_joint = 5
                    # Print out to user
                    print("New joint being tuned: {}".format(self.active_arm_joint))
            elif key.char == "6":
                # Make sure range is valid; if so, update this specific joint
                if self._check_valid_joint(6):
                    self.active_arm_joint = 6
                    # Print out to user
                    print("New joint being tuned: {}".format(self.active_arm_joint))
            elif key.char == "7":
                # Make sure range is valid; if so, update this specific joint
                if self._check_valid_joint(7):
                    self.active_arm_joint = 7
                    # Print out to user
                    print("New joint being tuned: {}".format(self.active_arm_joint))
            elif key.char == "8":
                # Make sure range is valid; if so, update this specific joint
                if self._check_valid_joint(8):
                    self.active_arm_joint = 8
                    # Print out to user
                    print("New joint being tuned: {}".format(self.active_arm_joint))
            elif key.char == "9":
                # Make sure range is valid; if so, update this specific joint
                if self._check_valid_joint(9):
                    self.active_arm_joint = 9
                    # Print out to user
                    print("New joint being tuned: {}".format(self.active_arm_joint))
            elif key.char == "t":
                # Toggle active arm
                self._toggle_arm()
            elif key.char == "r":
                # Reset active arm joint qpos to 0
                self.set_joint_positions(np.zeros(self.num_joints))

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.

        Args:
            key: [NOT USED]
        """
        pass

    def set_joint_positions(self, qpos):
        """
        Automatically sets the joint positions to be the given value

        Args:
            qpos (np.array): Joint positions to set
        """
        self.current_joints_pos = qpos
        self._update_joint_position(1, 0)

    def _check_valid_joint(self, i):
        """
        Checks to make sure joint number request @i is within valid range

        Args:
            i (int): Index to validate

        Returns:
            bool: True if index @i is valid, else prints out an error and returns False
        """
        if i > self.num_joints:
            # Print error
            print("Error: Requested joint {} is out of range; available joints are 1 - {}".format(i, self.num_joints))
            return False
        else:
            return True

    def _toggle_arm(self):
        """
        Toggle between arms in the environment to set as current active arm
        """
        if isinstance(self.active_robot, SingleArm):
            self.active_robot_num = (self.active_robot_num + 1) % self.num_robots
            robot = self.active_robot_num
        else:  # Bimanual case
            self.active_arm = "left" if self.active_arm == "right" else "right"
            robot = self.active_arm
        # Reset joint being controlled to 1
        self.active_arm_joint = 1
        # Print out new robot to user
        print("New robot arm being tuned: {}".format(robot))

    def _update_joint_position(self, i, delta):
        """
        Updates specified joint position @i by value @delta from its current position
        Note: assumes @i is already within the valid joint range

        Args:
            i (int): Joint index to update
            delta (float): Increment to alter specific joint by
        """
        self.current_joints_pos[i - 1] += delta
        if isinstance(self.active_robot, SingleArm):
            robot = self.active_robot_num
            self.env.sim.data.qpos[self.active_robot._ref_joint_pos_indexes] = self.current_joints_pos
        else:  # Bimanual case
            robot = self.active_arm
            if self.active_arm == "right":
                self.env.sim.data.qpos[
                    self.active_robot._ref_joint_pos_indexes[: self.num_joints]
                ] = self.current_joints_pos
            else:  # left arm case
                self.env.sim.data.qpos[
                    self.active_robot._ref_joint_pos_indexes[self.num_joints :]
                ] = self.current_joints_pos
        # Print out current joint positions to user
        print("Robot {} joint qpos: {}".format(robot, self.current_joints_pos))

    @property
    def active_robot(self):
        """
        Returns:
            Robot: active robot arm currently being tuned
        """
        return self.env.robots[self.active_robot_num]

    @property
    def num_joints(self):
        """
        Returns:
            int: number of joints for the current arm
        """
        if isinstance(self.active_robot, SingleArm):
            return len(self.active_robot.torque_limits[0])
        else:  # Bimanual arm case
            return int(len(self.active_robot.torque_limits[0]) / 2)


def print_command(char, info):
    """
    Prints out the command + relevant info entered by user

    Args:
        char (str): Command entered
        info (str): Any additional info to print
    """
    char += " " * (10 - len(char))
    print("{}\t{}".format(char, info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--init_qpos", nargs="+", type=float, default=0, help="Initial qpos to use. 0 defaults to all zeros"
    )

    args = parser.parse_args()

    print(
        "\nWelcome to the joint tuning script! You will be able to tune the robot\n"
        "arm joints in the specified environment by using your keyboard. The \n"
        "controls are printed below:"
    )

    print("")
    print_command("Keys", "Command")
    print_command("1-N", "Active Joint being tuned (N=number of joints for the active arm)")
    print_command("t", "Toggle between robot arms in the environment")
    print_command("r", "Reset active arm joints to all 0s")
    print_command("up/down", "incr/decrement the active joint angle")
    print_command("right/left", "incr/decrement the delta joint angle per up/down keypress")
    print("")

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # Define the controller
    controller_config = robosuite.load_controller_config(default_controller="JOINT_POSITION")

    # make the environment
    env = robosuite.make(
        args.env,
        robots=args.robots,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        render_camera=None,
        controller_configs=controller_config,
        initialization_noise=None,
    )
    env.reset()

    # register callbacks to handle key presses in the viewer
    key_handler = KeyboardHandler(env=env)

    # Set initial state
    if type(args.init_qpos) == int and args.init_qpos == 0:
        # Default to all zeros
        pass
    else:
        key_handler.set_joint_positions(args.init_qpos)

    # just spin to let user interact with window
    while True:
        action = np.zeros(env.action_dim)
        obs, reward, done, _ = env.step(action)
        env.render()
