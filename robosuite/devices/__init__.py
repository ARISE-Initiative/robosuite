from .device import Device

try:
    from .keyboard import Keyboard
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module pynput, required to interface with Keyboard. "
    ) from exc

try:
    from .spacemouse import SpaceMouse
except ImportError as e:
    print("Exception!", e)
    print(
        """Unable to load module hid, required to interface with SpaceMouse.\n
           Only macOS is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`"""
    )
