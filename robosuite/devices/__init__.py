from .device import Device
from .keyboard import Keyboard

try:
    from .spacemouse import SpaceMouse
    from .dualsense import DualSense
    from .logitech_gf310 import LogitechGF310
except ImportError as e:
    print("Exception!", e)
    print(
        """Unable to load module hid, required to interface with SpaceMouse or DualSense.\n
           Only macOS is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`"""
    )
