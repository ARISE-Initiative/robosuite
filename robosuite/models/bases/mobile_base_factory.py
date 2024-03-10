"""
Defines a string based method of initializing mounts
"""


def mobile_base_factory(name, idn=0):
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
    from robosuite.models.mobile_bases import MOBILE_BASE_MAPPING

    return MOBILE_BASE_MAPPING.get(name, "Unknown mobile base name: {}".format(name))(idn=idn)
