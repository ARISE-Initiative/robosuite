from .device import Device
from .keyboard import Keyboard
try:
    from .spacemouse import SpaceMouse
except ImportError:
    print("""Unable to load module hid, required to interface with SpaceMouse.\n
           Only Mac OS X is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`""")
