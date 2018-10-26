class robosuiteError(Exception):
    """Base class for exceptions in robosuite."""

    pass


class XMLError(robosuiteError):
    """Exception raised for errors related to xml."""

    pass


class SimulationError(robosuiteError):
    """Exception raised for errors during runtime."""

    pass


class RandomizationError(robosuiteError):
    """Exception raised for really really bad RNG."""

    pass
