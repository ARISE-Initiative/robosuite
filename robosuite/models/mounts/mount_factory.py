"""
Defines a string based method of initializing mounts
"""


def mount_factory(name, idn=0):
    """
    Generator for grippers

    Creates a MountModel instance with the provided name.

    Args:
        name (None or str): the name of the mount class
        idn (int or str): Number or some other unique identification string for this mount instance

    Returns:
        MountModel: requested mount instance

    Raises:
        XMLError: [invalid XML]
    """
    # Import MOUNT_MAPPING at runtime so we avoid circular imports
    from robosuite.models.mounts import MOUNT_MAPPING
    return MOUNT_MAPPING.get(name, "Unknown mount name: {}".format(name))(idn=idn)
