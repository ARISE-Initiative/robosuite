import numpy as np
from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements
from robosuite.models.objects import CylinderObject


class WipeArena(TableArena):
    """
    Workspace that contains an empty table with visual markers on its surface.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        coverage_factor (float): Fraction of table that will be sampled for dirt placement
        num_markers (int): Number of dirt (peg) particles to generate in a path on the table
        table_friction_std (float): Standard deviation to sample for the peg friction
        line_width (float): Diameter of dirt path trace
        two_clusters (bool): If set, will generate two separate dirt paths with half the number of sensors in each
    """

    def __init__(
        self, 
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(0.01, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        coverage_factor=0.9,
        num_markers=10,
        table_friction_std=0,
        line_width=0.02,
        two_clusters=False
    ):
        # Tactile table-specific features
        self.table_friction_std = table_friction_std
        self.line_width = line_width
        self.markers = []
        self.coverage_factor = coverage_factor
        self.num_markers = num_markers
        self.two_clusters = two_clusters

        # Attribute to hold current direction of sampled dirt path
        self.direction = None

        # run superclass init
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
        )

    def configure_location(self):
        """Configures correct locations for this arena"""
        # Run superclass first
        super().configure_location()

        # Define start position for drawing the line
        pos = self.sample_start_pos()

        # Define dirt material for markers
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.0",
            "shininess": "0.0",
        }
        dirt = CustomMaterial(
            texture="Dirt",
            tex_name="dirt",
            mat_name="dirt_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
            shared=True,
        )

        # Define line(s) drawn on table
        for i in range(self.num_markers):
            # If we're using two clusters, we resample the starting position and direction at the halfway point
            if self.two_clusters and i == int(np.floor(self.num_markers / 2)):
                pos = self.sample_start_pos()
            marker_name = f'contact{i}'
            marker = CylinderObject(
                name=marker_name,
                size=[self.line_width / 2, 0.001],
                rgba=[1, 1, 1, 1],
                material=dirt,
                obj_type="visual",
                joints=None,
            )
            # Manually add this object to the arena xml
            self.merge_assets(marker)
            table = find_elements(root=self.worldbody, tags="body", attribs={"name": "table"}, return_first=True)
            table.append(marker.get_obj())

            # Add this marker to our saved list of all markers
            self.markers.append(marker)

            # Add to the current dirt path
            pos = self.sample_path_pos(pos)

    def reset_arena(self, sim):
        """
        Reset the visual marker locations in the environment. Requires @sim (MjSim) reference to be passed in so that
        the Mujoco sim can be directly modified

        Args:
            sim (MjSim): Simulation instance containing this arena and visual markers
        """
        # Sample new initial position and direction for generated marker paths
        pos = self.sample_start_pos()

        # Loop through all visual markers
        for i, marker in enumerate(self.markers):
            # If we're using two clusters, we resample the starting position and direction at the halfway point
            if self.two_clusters and i == int(np.floor(self.num_markers / 2)):
                pos = self.sample_start_pos()
            # Get IDs to the body, geom, and site of each marker
            body_id = sim.model.body_name2id(marker.root_body)
            geom_id = sim.model.geom_name2id(marker.visual_geoms[0])
            site_id = sim.model.site_name2id(marker.sites[0])
            # Determine new position for this marker
            position = np.array([pos[0], pos[1], self.table_half_size[2]])
            # Set the current marker (body) to this new position
            sim.model.body_pos[body_id] = position
            # Reset the marker visualization -- setting geom rgba alpha value to 1
            sim.model.geom_rgba[geom_id][3] = 1
            # Hide the default visualization site
            sim.model.site_rgba[site_id][3] = 0
            # Sample next values in local marker trajectory
            pos = self.sample_path_pos(pos)

    def sample_start_pos(self):
        """
        Helper function to return sampled start position of a new dirt (peg) location

        Returns:
            np.array: the (x,y) value of the newly sampled dirt starting location
        """
        # First define the random direction that we will start at
        self.direction = np.random.uniform(-np.pi, np.pi)

        return np.array(
            (
                np.random.uniform(
                    -self.table_half_size[0] * self.coverage_factor + self.line_width / 2,
                    self.table_half_size[0] * self.coverage_factor - self.line_width / 2),
                np.random.uniform(
                    -self.table_half_size[1] * self.coverage_factor + self.line_width / 2,
                    self.table_half_size[1] * self.coverage_factor - self.line_width / 2)
            )
        )

    def sample_path_pos(self, pos):
        """
        Helper function to add a sampled dirt (peg) position to a pre-existing dirt path, whose most
        recent dirt position is defined by @pos

        Args:
            pos (np.array): (x,y) value of most recent dirt position

        Returns:
            np.array: the (x,y) value of the newly sampled dirt position to add to the current dirt path
        """
        # Random chance to alter the current dirt direction
        if np.random.uniform(0, 1) > 0.7:
            self.direction += np.random.normal(0, 0.5)

        posnew0 = pos[0] + 0.005 * np.sin(self.direction)
        posnew1 = pos[1] + 0.005 * np.cos(self.direction)

        # We keep resampling until we get a valid new position that's on the table
        while abs(posnew0) >= self.table_half_size[0] * self.coverage_factor - self.line_width / 2 or \
                abs(posnew1) >= self.table_half_size[1] * self.coverage_factor - self.line_width / 2:
            self.direction += np.random.normal(0, 0.5)
            posnew0 = pos[0] + 0.005 * np.sin(self.direction)
            posnew1 = pos[1] + 0.005 * np.cos(self.direction)

        # Return this newly sampled position
        return np.array((posnew0, posnew1))

