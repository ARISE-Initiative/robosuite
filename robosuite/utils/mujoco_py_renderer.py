from mujoco_py import MjViewer
from mujoco_py.generated import const
import glfw
from collections import defaultdict


class CustomMjViewer(MjViewer):
    """
    Custom class extending the vanilla MjViewer class to add additional key-stroke callbacks
    """

    keypress = defaultdict(list)
    keyup = defaultdict(list)
    keyrepeat = defaultdict(list)

    def key_callback(self, window, key, scancode, action, mods):
        """
        Processes key callbacks from the glfw renderer

        Args:
            window (GLFWwindow): GLFW window instance
            key (int): keycode
            scancode (int): scancode
            action (int): action code
            mods (int): mods
        """
        if action == glfw.PRESS:
            tgt = self.keypress
        elif action == glfw.RELEASE:
            tgt = self.keyup
        elif action == glfw.REPEAT:
            tgt = self.keyrepeat
        else:
            return
        if tgt.get(key):
            for fn in tgt[key]:
                fn(window, key, scancode, action, mods)
        if tgt.get("any"):
            for fn in tgt["any"]:
                fn(window, key, scancode, action, mods)
            # retain functionality for closing the viewer
            if key == glfw.KEY_ESCAPE:
                super().key_callback(window, key, scancode, action, mods)
        else:
            # only use default mujoco callbacks if "any" callbacks are unset
            super().key_callback(window, key, scancode, action, mods)


class MujocoPyRenderer:
    """
    Mujoco-py renderer object

    Args:
        sim: MjSim object
    """

    def __init__(self, sim):
        self.viewer = CustomMjViewer(sim)
        self.callbacks = {}

    def set_camera(self, camera_id):
        """
        Set the camera view to the specified camera ID.

        Args:
            camera_id (int): id of the camera to set the current viewer to
        """
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.type = const.CAMERA_FIXED

    def render(self):
        """
        Renders the screen
        """
        # safe for multiple calls
        self.viewer.render()

    def close(self):
        """
        Destroys the open window and renders (pun intended) the viewer useless.
        """
        glfw.destroy_window(self.viewer.window)
        self.viewer = None

    def add_keypress_callback(self, key, fn):
        """
        Allows for custom callback functions for the viewer. Called on key down.
        Parameter 'any' will ensure that the callback is called on any key down,
        and block default mujoco viewer callbacks from executing, except for
        the ESC callback to close the viewer.

        Args:
            key (int): keycode
            fn (function handle): function callback to associate with the keypress
        """
        self.viewer.keypress[key].append(fn)

    def add_keyup_callback(self, key, fn):
        """
        Allows for custom callback functions for the viewer. Called on key up.
        Parameter 'any' will ensure that the callback is called on any key up,
        and block default mujoco viewer callbacks from executing, except for
        the ESC callback to close the viewer.

        Args:
            key (int): keycode
            fn (function handle): function callback to associate with the keypress
        """
        self.viewer.keyup[key].append(fn)

    def add_keyrepeat_callback(self, key, fn):
        """
        Allows for custom callback functions for the viewer. Called on key repeat.
        Parameter 'any' will ensure that the callback is called on any key repeat,
        and block default mujoco viewer callbacks from executing, except for
        the ESC callback to close the viewer.

        Args:
            key (int): keycode
            fn (function handle): function callback to associate with the keypress
        """
        self.viewer.keyrepeat[key].append(fn)
