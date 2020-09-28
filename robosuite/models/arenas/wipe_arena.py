import numpy as np
from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import array_to_string, CustomMaterial
from robosuite.models.objects import CylinderObject


class WipeArena(TableArena):
    """
    Workspace that contains an empty table with tactile sensors on its surface.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        coverage_factor (float): Fraction of table that will be sampled for dirt placement
        num_sensors (int): Number of dirt (peg) particles to generate in a path on the table
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
        num_sensors=10,
        table_friction_std=0,
        line_width=0.02,
        two_clusters=False
    ):
        # Tactile table-specific features
        self.table_friction_std = table_friction_std
        self.line_width = line_width
        self.sensor_names = []
        self.sensor_site_names = {}
        self.coverage_factor = coverage_factor
        self.num_sensors = num_sensors
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

        # Determine peg friction
        friction = max(0.001, np.random.normal(self.table_friction[0], self.table_friction_std))

        # Grab reference to the table body in the xml
        table_subtree = self.worldbody.find(".//body[@name='{}']".format("table"))

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
        )

        # Define line(s) drawn on table
        for i in range(self.num_sensors):
            # If we're using two clusters, we resample the starting position and direction at the halfway point
            if self.two_clusters and i == int(np.floor(self.num_sensors / 2)):
                pos = self.sample_start_pos()
            square_name2 = 'contact_'+str(i)
            square2 = CylinderObject(
                name=square_name2,
                size=[self.line_width / 2, 0.001],
                rgba=[1, 1, 1, 1],
                density=1,
                material=dirt,
                friction=friction,
            )
            self.merge_asset(square2)
            visual_c = square2.get_visual(site=True)
            visual_c.set("pos", array_to_string([pos[0], pos[1], self.table_half_size[2]]))
            visual_c.find("site").set("pos", [0, 0, 0.005])
            visual_c.find("site").set("rgba", array_to_string([0, 0, 0, 0]))
            table_subtree.append(visual_c)

            sensor_name = square_name2 + "_sensor"
            sensor_site_name = square_name2
            self.sensor_names += [sensor_name]
            self.sensor_site_names[sensor_name] = sensor_site_name

            # Add to the current dirt path
            pos = self.sample_path_pos(pos)

    def reset_arena(self, sim):
        """
        Reset the tactile sensor locations in the environment. Requires @sim (MjSim) reference to be passed in so that
        the Mujoco sim can be directly modified

        Args:
            sim (MjSim): Simulation instance containing this arena and tactile sensors
        """
        # Sample new initial position and direction for generated sensor paths
        pos = self.sample_start_pos()

        # Loop through all sensor collision body / site pairs
        for i, (_, sensor_name) in enumerate(self.sensor_site_names.items()):
            # If we're using two clusters, we resample the starting position and direction at the halfway point
            if self.two_clusters and i == int(np.floor(self.num_sensors / 2)):
                pos = self.sample_start_pos()
            # Get IDs to the body, geom, and site of each sensor
            body_id = sim.model.body_name2id(sensor_name)
            geom_id = sim.model.geom_name2id(sensor_name)
            # Determine new position for this sensor
            position = np.array([pos[0], pos[1], self.table_half_size[2]])
            # Set the current sensor (body) to this new position
            sim.model.body_pos[body_id] = position
            # Reset the sensor visualization -- setting geom rgba to all 1's
            sim.model.geom_rgba[geom_id] = [1, 1, 1, 1]
            # Sample next values in local sensor trajectory
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

