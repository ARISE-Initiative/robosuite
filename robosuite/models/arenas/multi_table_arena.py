from collections import Iterable

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import (
    array_to_string,
    new_body,
    new_geom,
    new_site,
    string_to_array,
    xml_path_completion,
)


class MultiTableArena(Arena):
    """
    Workspace that contains multiple tables.
    Args:
        table_offsets (list of 3-array): (x,y,z) offset from center of arena when placing each table.
            Note that the number of tables is inferred from the length of this list
            Note that the z value sets the upper limit of the table
        table_rots (float or list of float): z-rotation to apply to each table. If only a
            single value is given, it will be broadcasted according to the total number of tables
        table_full_sizes (3-array or list of 3-array): (L,W,H) full dimensions of each table. If only a
            single value is given, it will be broadcasted according to the total number of tables
        table_frictions (3-array or list of 3-array): (sliding, torsional, rolling) friction parameters of each table.
        has_legs (bool or list of bool): whether each table has legs or not. If only a
            single value is given, it will be broadcasted according to the total number of tables
        xml (str): xml file to load arena
    """

    def __init__(
        self,
        table_offsets,
        table_rots=0,
        table_full_sizes=(0.8, 0.8, 0.05),
        table_frictions=(1, 0.005, 0.0001),
        has_legs=True,
        xml="arenas/multi_table_arena.xml",
    ):
        # Set internal vars
        self.table_offsets = np.array(table_offsets)
        self.n_tables = self.table_offsets.shape[0]
        self.table_rots = (
            np.array(table_rots) if isinstance(table_rots, Iterable) else np.ones(self.n_tables) * table_rots
        )
        self.table_full_sizes = np.array(table_full_sizes)
        if len(self.table_full_sizes.shape) == 1:
            self.table_full_sizes = np.stack([self.table_full_sizes] * self.n_tables, axis=0)
        self.table_half_sizes = self.table_full_sizes / 2
        self.table_frictions = np.array(table_frictions)
        if len(self.table_frictions.shape) == 1:
            self.table_frictions = np.stack([self.table_frictions] * self.n_tables, axis=0)
        self.center_pos = np.array(self.table_offsets)
        self.center_pos[:, 2] -= self.table_half_sizes[:, 2]
        self.has_legs = has_legs if isinstance(has_legs, Iterable) else [has_legs] * self.n_tables

        # Run super init
        super().__init__(xml_path_completion(xml))

        # Configure any relevant locations
        self.configure_location()

    def _add_table(self, name, offset, rot, half_size, friction, has_legs):
        """
        Procedurally generates a table and adds it to the XML
        """
        # Create body for this table, and add it to worldbody
        table_body = new_body(name=name, pos=offset - np.array([0, 0, half_size[2]]))
        self.worldbody.append(table_body)

        # Create core attributes for table geoms
        table_attribs = {
            "pos": (0, 0, 0),
            "quat": T.convert_quat(T.axisangle2quat([0, 0, rot]), to="wxyz"),
            "size": half_size,
            "type": "box",
        }

        # Create collision and visual bodies, and add them to the table body
        col_geom = new_geom(name=f"{name}_collision", group=0, friction=friction, **table_attribs)
        vis_geom = new_geom(
            name=f"{name}_visual", group=1, conaffinity=0, contype=0, material="table_ceramic", **table_attribs
        )
        table_body.append(col_geom)
        table_body.append(vis_geom)

        # Add tabletop site to table
        top_site = new_site(name=f"{name}_top", pos=(0, 0, half_size[2]), size=(0.001, 0.001, 0.001), rgba=(0, 0, 0, 0))
        table_body.append(top_site)

        # Add legs if requested
        if has_legs:
            delta_x = [0.1, -0.1, -0.1, 0.1]
            delta_y = [0.1, 0.1, -0.1, -0.1]
            for i, (dx, dy) in enumerate(zip(delta_x, delta_y)):
                # If x-length of table is less than a certain length, place leg in the middle between ends
                # Otherwise we place it near the edge
                x = 0
                if half_size[0] > abs(dx * 2.0):
                    x += np.sign(dx) * half_size[0] - dx
                # Repeat the same process for y
                y = 0
                if half_size[1] > abs(dy * 2.0):
                    y += np.sign(dy) * half_size[1] - dy
                # Rotate x and y values according to requested rotation
                c, s = np.cos(rot), np.sin(rot)
                rot_xy = np.array([[c, -s], [s, c]]) @ np.array([x, y])
                # Add in offsets
                x = rot_xy[0]
                y = rot_xy[1]
                # Get z value
                z = (offset[2] - half_size[2]) / 2.0
                # Create visual geom and add it to table body
                leg_geom = new_geom(
                    name=f"{name}_leg{i}_visual",
                    pos=(x, y, -z),
                    type="cylinder",
                    size=(0.025, z),
                    group=1,
                    conaffinity=0,
                    contype=0,
                    material="table_legs_metal",
                )
                table_body.append(leg_geom)

    def configure_location(self):
        """Configures correct locations for this arena"""
        # Set floor correctly
        self.floor.set("pos", array_to_string(self.bottom_pos))

    def _postprocess_arena(self):
        """
        Runs any necessary post-processing on the imported Arena model
        """
        # Create tables
        for i, (offset, rot, half_size, friction, legs) in enumerate(
            zip(self.table_offsets, self.table_rots, self.table_half_sizes, self.table_frictions, self.has_legs)
        ):
            self._add_table(
                name=f"table{i}",
                offset=offset,
                rot=rot,
                half_size=half_size,
                friction=friction,
                has_legs=legs,
            )
