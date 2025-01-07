"""Driver class for SpaceMouse controller.

This class provides a driver support to SpaceMouse on macOS.
In particular, we assume you are using a SpaceMouse Wireless by default.

To set up a new SpaceMouse controller:
    1. Download and install driver from https://www.3dconnexion.com/service/drivers.html
    2. Install hidapi library through pip
       (make sure you run uninstall hid first if it is installed).
    3. Make sure SpaceMouse is connected before running the script
    4. (Optional) Based on the model of SpaceMouse, you might need to change the
       vendor id and product id that correspond to the device.

For Linux support, you can find open-source Linux drivers and SDKs online.
    See http://spacenav.sourceforge.net/

"""

import threading
import time
from collections import namedtuple

import numpy as np
try:
    from pynput.keyboard import Controller, Key, Listener
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module pynput, required to interface with Keyboard. "
    ) from exc

from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

try:
    import hid
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module hid, required to interface with SpaceMouse. "
        "Only macOS is officially supported. Install the additional "
        "requirements with `pip install -r requirements-extra.txt`"
    ) from exc

from pynput.keyboard import Controller, Key, Listener

import robosuite.macros as macros
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix

AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}


def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte

    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.

    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling

    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.

    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte

    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


class SpaceMouse(Device):
    """
    A minimalistic driver class for SpaceMouse with HID library.

    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.
    You can look up its vendor/product id from this method.

    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(
        self,
        env,
        vendor_id=macros.SPACEMOUSE_VENDOR_ID,
        product_id=macros.SPACEMOUSE_PRODUCT_ID,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
    ):
        super().__init__(env)

        print("Opening SpaceMouse device")
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = hid.device()
        try:
            self.device.open(self.vendor_id, self.product_id)  # SpaceMouse
        except OSError as e:
            ROBOSUITE_DEFAULT_LOGGER.warning(
                "Failed to open SpaceMouse device"
                "Consider killing other processes that may be using the device such as 3DconnexionHelper (killall 3DconnexionHelper)"
            )
            raise

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

        # also add a keyboard for aux controls
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Right button", "reset simulation")
        print_command("Left button (hold)", "close gripper")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print_command("Control+C", "quit")
        print_command("b", "toggle arm/base mode (if applicable)")
        print_command("s", "switch active arm (if multi-armed robot)")
        print_command("=", "switch active robot (if multi-robot environment)")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        super()._reset_internal_state()

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )

    def run(self):
        """Listener method that keeps pulling new messages."""

        t_last_click = -1

        while True:
            d = self.device.read(13)
            if d is not None and self._enabled:

                if self.product_id == 50741:
                    ## logic for older spacemouse model

                    if d[0] == 1:  ## readings from 6-DoF sensor
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0

                    elif d[0] == 2:

                        self.roll = convert(d[1], d[2])
                        self.pitch = convert(d[3], d[4])
                        self.yaw = convert(d[5], d[6])

                        self._control = [
                            self.x,
                            self.y,
                            self.z,
                            self.roll,
                            self.pitch,
                            self.yaw,
                        ]
                else:
                    ## default logic for all other spacemouse models

                    if d[0] == 1:  ## readings from 6-DoF sensor
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0

                        self.roll = convert(d[7], d[8])
                        self.pitch = convert(d[9], d[10])
                        self.yaw = convert(d[11], d[12])

                        self._control = [
                            self.x,
                            self.y,
                            self.z,
                            self.roll,
                            self.pitch,
                            self.yaw,
                        ]

                if d[0] == 3:  ## readings from the side buttons

                    # press left button
                    if d[1] == 1:
                        t_click = time.time()
                        elapsed_time = t_click - t_last_click
                        t_last_click = t_click
                        self.single_click_and_hold = True

                    # release left button
                    if d[1] == 0:
                        self.single_click_and_hold = False

                    # right button is for reset
                    if d[1] == 2:
                        self._reset_state = 1
                        self._enabled = False
                        self._reset_internal_state()

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse

        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.

        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """
        pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        try:
            # controls for mobile base (only applicable if mobile base present)
            if key.char == "b":
                self.base_modes[self.active_robot] = not self.base_modes[self.active_robot]  # toggle mobile base
            elif key.char == "s":
                self.active_arm_index = (self.active_arm_index + 1) % len(self.all_robot_arms[self.active_robot])
            elif key.char == "=":
                self.active_robot = (self.active_robot + 1) % self.num_robots

        except AttributeError as e:
            pass

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 50
        dpos = dpos * 125

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation


if __name__ == "__main__":

    space_mouse = SpaceMouse()
    for i in range(100):
        print(space_mouse.control, space_mouse.control_gripper)
        time.sleep(0.02)
