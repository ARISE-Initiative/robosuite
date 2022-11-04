"""
Macro settings that can be imported and toggled. Internally, specific parts of the codebase rely on these settings
for determining core functionality.

To make sure global reference is maintained, should import these settings as:

`import robosuite.macros as macros`
"""

# Global Mujoco Simulation Parameters
SIMULATION_TIMESTEP = 0.002  # Internal simulation timestep (in seconds)

# Instance Randomization
# Used if we want to randomize geom groups uniformly per instance -- e.g.: entire robot arm, vs. per-joint geom
# This should get set to True in your script BEFORE an environment is created or the DR wrapper is used
USING_INSTANCE_RANDOMIZATION = False

# Numba settings
# TODO: Numba causes BSOD for NutAssembly task when rendering offscreen (deterministically!)
ENABLE_NUMBA = True
CACHE_NUMBA = True

# Image Convention
# Robosuite (Mujoco)-rendered images are based on the OpenGL coordinate frame convention, whereas many downstream
# applications assume an OpenCV coordinate frame convention. For consistency, you can set the image convention
# here; this will assure that any rendered frames will match the associated convention.
# See the figure at the bottom of https://amytabb.com/ts/2019_06_28/ for an informative overview.
IMAGE_CONVENTION = "opengl"  # Options are {"opengl", "opencv"}

# Image concatenation
# In general, observations are concatenated together by modality. However, image observations are expensive memory-wise,
# so we skip concatenating all images together by default, unless this flag is set to True
CONCATENATE_IMAGES = False

MUJOCO_GPU_RENDERING = True

# Spacemouse settings. Used by SpaceMouse class in robosuite/devices/spacemouse.py
SPACEMOUSE_VENDOR_ID = 9583
SPACEMOUSE_PRODUCT_ID = 50735


try:
    from robosuite.macros_private import *
except ImportError:
    from robosuite.utils.log_utils import log_warning

    log_warning(
        "No private macro file found!"
        "\nIt is recommended to use a private macro file"
        "\nTo setup, run: python robosuite/scripts/setup_macros.py"
    )
