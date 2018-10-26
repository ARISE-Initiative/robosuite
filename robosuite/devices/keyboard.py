"""
Driver class for Keyboard controller.
"""

import glfw
import numpy as np
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class Keyboard(Device):
    """A minimalistic driver class for a Keyboard."""

    def __init__(self):
        """
        Initialize a Keyboard device.
        """

        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False
        self._pos_step = 0.05

    def _display_controls(self):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("q", "reset simulation")
        print_command("spacebar", "toggle gripper (open/close)")
        print_command("w-a-s-d", "move arm horizontally in x-y plane")
        print_command("r-f", "move arm vertically")
        print_command("z-x", "rotate arm about x-axis")
        print_command("t-g", "rotate arm about y-axis")
        print_command("c-v", "rotate arm about z-axis")
        print_command("ESC", "quit")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self.grasp = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """Returns the current state of the keyboard, a dictionary of pos, orn, grasp, and reset."""
        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
        )

    def on_press(self, window, key, scancode, action, mods):
        """
        Key handler for key presses.
        """

        # controls for moving position
        if key == glfw.KEY_W:
            self.pos[0] -= self._pos_step  # dec x
        elif key == glfw.KEY_S:
            self.pos[0] += self._pos_step  # inc x
        elif key == glfw.KEY_A:
            self.pos[1] -= self._pos_step  # dec y
        elif key == glfw.KEY_D:
            self.pos[1] += self._pos_step  # inc y
        elif key == glfw.KEY_F:
            self.pos[2] -= self._pos_step  # dec z
        elif key == glfw.KEY_R:
            self.pos[2] += self._pos_step  # inc z

        # controls for moving orientation
        elif key == glfw.KEY_Z:
            drot = rotation_matrix(angle=0.1, direction=[1., 0., 0.])[:3, :3]
            self.rotation = self.rotation.dot(drot)  # rotates x
        elif key == glfw.KEY_X:
            drot = rotation_matrix(angle=-0.1, direction=[1., 0., 0.])[:3, :3]
            self.rotation = self.rotation.dot(drot)  # rotates x
        elif key == glfw.KEY_T:
            drot = rotation_matrix(angle=0.1, direction=[0., 1., 0.])[:3, :3]
            self.rotation = self.rotation.dot(drot)  # rotates y
        elif key == glfw.KEY_G:
            drot = rotation_matrix(angle=-0.1, direction=[0., 1., 0.])[:3, :3]
            self.rotation = self.rotation.dot(drot)  # rotates y
        elif key == glfw.KEY_C:
            drot = rotation_matrix(angle=0.1, direction=[0., 0., 1.])[:3, :3]
            self.rotation = self.rotation.dot(drot)  # rotates z
        elif key == glfw.KEY_V:
            drot = rotation_matrix(angle=-0.1, direction=[0., 0., 1.])[:3, :3]
            self.rotation = self.rotation.dot(drot)  # rotates z

    def on_release(self, window, key, scancode, action, mods):
        """
        Key handler for key releases.
        """

        # controls for grasping
        if key == glfw.KEY_SPACE:
            self.grasp = not self.grasp  # toggle gripper

        # user-commanded reset
        elif key == glfw.KEY_Q:
            self._reset_state = 1
            self._enabled = False
            self._reset_internal_state()
