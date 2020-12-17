"""
Macro settings that can be imported and toggled. Internally, specific parts of the codebase rely on these settings
for determining core functionality.

To make sure global reference is maintained, should import these settings as:

`import robosuite.utils.macros as macros`
"""

# Global Mujoco Simulation Parameters
SIMULATION_TIMESTEP = 0.002     # Internal simulation timestep (in seconds)

# Instance Randomization
# Used if we want to randomize geom groups uniformly per instance -- e.g.: entire robot arm, vs. per-joint geom
# This should get set to True in your script BEFORE an environment is created or the DR wrapper is used
USING_INSTANCE_RANDOMIZATION = False

# Numba settings
ENABLE_NUMBA = True
CACHE_NUMBA = True

# Image Convention
# Robosuite (Mujoco)-rendered images are based on the OpenGL coordinate frame convention, whereas many downstream
# applications assume an OpenCV coordinate frame convention. For consistency, you can set the image convention
# here; this will assure that any rendered frames will match the associated convention.
# See the figure at the bottom of https://amytabb.com/ts/2019_06_28/ for an informative overview.
IMAGE_CONVENTION = "opengl"     # Options are {"opengl", "opencv"}
