from robosuite.models.arenas import TableArena


class PegsArena(TableArena):
    """Workspace that contains a tabletop with two fixed pegs."""

    def __init__(
        self,
        table_full_size=(0.45, 0.69, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0),
    ):
        """
        Args:
            table_full_size: full dimensions of the table
            table_friction: friction parameters of the table
            table_offset: offset from center of arena when placing table
                Note that the z value sets the upper limit of the table
        """
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
            xml="arenas/pegs_arena.xml",
        )

        # Get references to peg bodies
        self.peg1_body = self.worldbody.find("./body[@name='peg1']")
        self.peg2_body = self.worldbody.find("./body[@name='peg2']")
