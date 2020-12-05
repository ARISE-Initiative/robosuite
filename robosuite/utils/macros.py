"""
Macro settings that can be imported and toggled. Internally, specific parts of the codebase rely on these settings
for determining core functionality.

To make sure global reference is maintained, should import these settings as:

`import robosuite.utils.macros as macros`
"""

# Setting for whether we're using Domain Randomization. This should get set to True in your script
# BEFORE an environment is created or the DR wrapper is used
USING_DOMAIN_RANDOMIZATION = False

# Numba settings
ENABLE_NUMBA = True
CACHE_NUMBA = True
