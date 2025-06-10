"""Driver class for Logitech GF310 controller.

This class provides a driver support to Logitech GF310 on macOS.
In particular, we assume you are using a wired connection by default.

"""

import threading
import time
from enum import IntFlag

import numpy as np

from robosuite import make
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

try:
    import hid
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module hid, required to interface with Logitech GF310. "
        "Only macOS is officially supported. Install the additional "
        "requirements with `pip install -r requirements-extra.txt`"
    ) from exc


import robosuite.macros as macros
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class ConnectionType(IntFlag):
    USB = 0x1
    BT01 = 0x2
    BT31 = 0x4
    UNKNOWN = 0x8

    @classmethod
    def to_string(cls, value) -> str:
        if value & cls.UNKNOWN:
            return "Unknown"
        if value & cls.BT01:
            return "Bluetooth 01"
        if value & cls.BT31:
            return "Bluetooth 31"
        if value & cls.USB:
            return "USB"
        return "Unknown"


USB_REPORT_LENGTH = 8
DUALSENSE_AXIS_LIST = ["LX", "LY", "RX", "RY", "L2_Trigger", "R2_Trigger"]
DUALSENSE_BTN_LIST = [
    "Y",  # Triangle
    "B",  # Circle
    "A",  # Cross
    "X",  # Square
    "DpadUp",
    "DpadDown",
    "DpadLeft",
    "DpadRight",
    "L1",
    "L2",
    "R1",
    "R2",
]
DUALSENSE_AXIS_MIN = 0
DUALSENSE_AXIS_MAX = 255
DUALSENSE_STICK_Neutral = 128
DUALSENSE_Trigger_Neutral = 0


def scale_to_control(x, axis_scale=128, min_v=-1.0, max_v=1.0):
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


class DSState:
    def __init__(self) -> None:
        """
        All dualsense states (inputs) that can be read. Second method to check if a input is pressed.
        """
        self.X, self.Y, self.B, self.A = False, False, False, False
        self.DpadUp, self.DpadDown, self.DpadLeft, self.DpadRight = (
            False,
            False,
            False,
            False,
        )
        self.L1, self.L2, self.R1, self.R2 = (
            False,
            False,
            False,
            False,
        )
        # neutral: 0x00, pressed: 0xff
        self.L2_Trigger, self.R2_Trigger = 0, 0
        self.RX, self.RY, self.LX, self.LY = 128, 128, 128, 128

    def __str__(self):
        return f"X: {self.X}, Y: {self.Y}, B: {self.B}, A: {self.A}, DpadUp: {self.DpadUp}, DpadDown: {self.DpadDown}, DpadLeft: {self.DpadLeft}, DpadRight: {self.DpadRight}, L1: {self.L1}, L2: {self.L2}, R1: {self.R1}, R2: {self.R2}"

    def setDPadState(self, dpad_state: int) -> None:
        """
        Sets the dpad state variables according to the integers that was read from the controller

        Args:
            dpad_state (int): integer number representing the dpad state, actually a 4-bit number,[0,8]
        """
        if dpad_state == 0:
            self.DpadUp = True
            self.DpadDown = False
            self.DpadLeft = False
            self.DpadRight = False
        elif dpad_state == 1:
            self.DpadUp = True
            self.DpadDown = False
            self.DpadLeft = False
            self.DpadRight = True
        elif dpad_state == 2:
            self.DpadUp = False
            self.DpadDown = False
            self.DpadLeft = False
            self.DpadRight = True
        elif dpad_state == 3:
            self.DpadUp = False
            self.DpadDown = True
            self.DpadLeft = False
            self.DpadRight = True
        elif dpad_state == 4:
            self.DpadUp = False
            self.DpadDown = True
            self.DpadLeft = False
            self.DpadRight = False
        elif dpad_state == 5:
            self.DpadUp = False
            self.DpadDown = True
            self.DpadLeft = True
            self.DpadRight = False
        elif dpad_state == 6:
            self.DpadUp = False
            self.DpadDown = False
            self.DpadLeft = True
            self.DpadRight = False
        elif dpad_state == 7:
            self.DpadUp = True
            self.DpadDown = False
            self.DpadLeft = True
            self.DpadRight = False
        else:
            self.DpadUp = False
            self.DpadDown = False
            self.DpadLeft = False
            self.DpadRight = False


class LogitechGF310(Device):
    """
    A minimalistic driver class for DualSense with HID library.

    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure DualSense is detected before running the script.
    You can look up its vendor/product id from this method.

    You can test your DualSense in https://hardwaretester.com/gamepad and https://nondebug.github.io/dualsense/dualsense-explorer.html
    DualSense HID protocol refer to https://github.com/nondebug/dualsense

    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
        reverse_xy (bool): Whether to reverse the effect of the x and y axes of the joystick. It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)
    """

    def __init__(
        self,
        env,
        vendor_id=macros.LOGITECH_GF310_VENDOR_ID,
        product_id=macros.LOGITECH_GF310_PRODUCT_ID,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        reverse_xy=False,
    ):
        super().__init__(env)

        print("Opening Logitech device")
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = hid.device()

        try:
            self.device.open(self.vendor_id, self.product_id)
        except Exception as e:
            ROBOSUITE_DEFAULT_LOGGER.warning(
                "Failed to open Logitech device. "
                "Consider killing other processes that may be using the device, "
                f"or try other product ids for SONY DualSense in {[hex(id) for id in macros.DUALSENSE_PRODUCT_IDs]}"
            )
            raise e

        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        self.input_report_length = -1
        self.output_report_length = -1
        self.connection_type = self._check_connection_type()
        print("Connection type: %s" % ConnectionType.to_string(self.connection_type))
        print("")
        print(
            "PS: You can modify `reverse_xy` if the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)."
        )

        self.reverse_xy = reverse_xy
        self.report_bytes: bytearray = None
        self.state: DSState = DSState()
        self.last_state: DSState = None

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        # launch a new listener thread to listen to DualSense
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (35 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Move LX/LY left joystick", "move arm horizontally in x-y plane")
        print_command("Press L2 Trigger with or without L1", "move arm vertically")
        print_command("Move RX/RY right joystick", "rotate arm about x/y axis, namely roll/pitch")
        print_command("Press R2 Trigger with or without R1", "rotate arm about z axis, namely yaw")
        print_command("X button", "reset simulation")
        print_command("B button (hold)", "close gripper")
        print_command("Y button", "toggle arm/base mode (if applicable)")
        print_command("Left/Right Direction Pad", "switch active arm (if multi-armed robot)")
        print_command("Up/Down Direction Pad", "switch active robot (if multi-robot environment)")
        print_command("Control+C", "quit")
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

    def _check_connection_type(self):
        """
        Get the connection type of the DualSense controller.
        ConnectionType:
        - USB: DualSense connected via USB
        - BT01: DualSense connected via Bluetooth, sends input report id 0x01
        - BT31: DualSense connected via Bluetooth, sends input report id 0x31
        - UNKNOWN: Unknown connection type

        Returns:
            ConnectionType: connection type(USB, BT01, BT31, UNKNOWN)
        """
        dummy_report = self.device.read(100)
        dummy_report_length = len(dummy_report)
        if dummy_report_length == USB_REPORT_LENGTH:
            self.input_report_length = USB_REPORT_LENGTH
            self.output_report_length = USB_REPORT_LENGTH
            return ConnectionType.USB
        return ConnectionType.UNKNOWN

    def _check_btn_changed(self, btn_name: str):
        """
        Check if a button has been pressed or released.

        Args:
            btn_name (str): Name of the button to check

        Returns:
            bool: True if the button has been pressed or released, False otherwise
        """
        assert btn_name in DUALSENSE_BTN_LIST
        return getattr(self.state, btn_name) != getattr(self.last_state, btn_name)

    def run(self):
        """Listener method that keeps pulling new messages."""
        t_last_click = -1

        while True:
            d = self.device.read(self.input_report_length)
            if d is not None and self._enabled:
                report_bytes = bytearray(d)
                self.report_bytes = report_bytes
                self.last_state = self.state

                if self.connection_type == ConnectionType.USB:
                    self.state = parse_usb_report(report_bytes)
                else:
                    raise NotImplementedError(f"Connection type {self.connection_type} not supported")
                self.x = scale_to_control(self.state.LX if not self.reverse_xy else self.state.LY)
                self.y = scale_to_control(self.state.LY if not self.reverse_xy else self.state.LX)
                self.roll = scale_to_control(self.state.RX if not self.reverse_xy else self.state.RY)
                self.pitch = scale_to_control(self.state.RY if not self.reverse_xy else self.state.RX)
                self.z = scale_to_control(self.state.L2_Trigger)
                if self.state.L1:
                    self.z = -self.z
                self.yaw = scale_to_control(self.state.R2_Trigger)
                if self.state.R1:
                    self.yaw = -self.yaw
                self._control = [
                    self.x,
                    self.y,
                    self.z,
                    self.roll,
                    self.pitch,
                    self.yaw,
                ]

                # press left button
                if self._check_btn_changed("B") and self.state.B:
                    t_click = time.time()
                    elapsed_time = t_click - t_last_click
                    t_last_click = t_click
                    self.single_click_and_hold = True

                # release left button
                if self._check_btn_changed("B") and not self.state.B:
                    self.single_click_and_hold = False

                # Reset
                if self._check_btn_changed("X") and self.state.X:
                    self._reset_state = 1
                    self._enabled = False
                    self._reset_internal_state()
                # controls for mobile base (only applicable if mobile base present)
                if self._check_btn_changed("Y") and self.state.Y:
                    self.base_modes[self.active_robot] = not self.base_modes[self.active_robot]  # toggle mobile base

                if self._check_btn_changed("DpadRight") and self.state.DpadRight:
                    self.active_arm_index = (self.active_arm_index + 1) % len(self.all_robot_arms[self.active_robot])
                elif self._check_btn_changed("DpadLeft") and self.state.DpadLeft:
                    self.active_arm_index = (self.active_arm_index - 1) % len(self.all_robot_arms[self.active_robot])

                if self._check_btn_changed("DpadUp") and self.state.DpadUp:
                    self.active_robot = (self.active_robot + 1) % self.num_robots
                if self._check_btn_changed("DpadDown") and self.state.DpadDown:
                    self.active_robot = (self.active_robot - 1) % self.num_robots

    @property
    def control(self):
        """
        Grabs current pose of DualSense

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

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 50
        dpos = dpos * 125

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation


def parse_usb_report(state_bytes: bytearray) -> DSState:
    new_state = DSState()

    new_state.LX = state_bytes[0] - 128
    new_state.LY = state_bytes[1] - 128
    new_state.RX = state_bytes[2] - 128
    new_state.RY = state_bytes[3] - 128
    new_state.L2_Trigger = state_bytes[5]
    new_state.R2_Trigger = state_bytes[6]

    buttonState = state_bytes[4]
    new_state.Y = (buttonState & (1 << 7)) != 0
    new_state.B = (buttonState & (1 << 6)) != 0
    new_state.A = (buttonState & (1 << 5)) != 0
    new_state.X = (buttonState & (1 << 4)) != 0

    # dpad
    dpad_state = buttonState & 0x0F
    new_state.setDPadState(dpad_state)

    misc = state_bytes[5]
    new_state.L1 = (misc & (1 << 0)) != 0
    new_state.R1 = (misc & (1 << 1)) != 0
    new_state.L2 = (misc & (1 << 2)) != 0
    new_state.R2 = (misc & (1 << 3)) != 0

    return new_state


if __name__ == "__main__":
    env = make("Lift", robots="Panda")
    dualsense = LogitechGF310(env)
    dualsense.start_control()
    for i in range(100):
        # print(dualsense.control, dualsense.control_gripper)
        print(dualsense.state)
        time.sleep(0.5)
