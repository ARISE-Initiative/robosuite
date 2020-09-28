from robosuite.models.arenas import TableArena


class PegsArena(TableArena):
    """
    Workspace that contains a tabletop with two fixed pegs.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
    """

    def __init__(
        self,
        table_full_size=(0.45, 0.69, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0),
    ):
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
            xml="arenas/pegs_arena.xml",
        )

        # Get references to peg bodies
        self.peg1_body = self.worldbody.find("./body[@name='peg1']")
        self.peg2_body = self.worldbody.find("./body[@name='peg2']")
