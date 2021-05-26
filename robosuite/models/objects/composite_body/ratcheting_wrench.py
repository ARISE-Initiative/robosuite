from robosuite.models.objects import CompositeBodyObject, BoxObject, CylinderObject, HollowCylinderObject
import numpy as np

from robosuite.utils.mjcf_utils import CustomMaterial


class RatchetingWrenchObject(CompositeBodyObject):
    """
    A ratcheting wrench made out of mujoco primitives.

    Args:
        name (str): Name of this object

        handle_size (float): (L, W, H) half-sizes for the handle (center part of wrench)

        outer_radius_1 (float): Outer radius of first end of wrench

        inner_radius_1 (float): Inner radius of first end of wrench

        height_1 (float): Height of first end of wrench

        outer_radius_2 (float): Outer radius of second end of wrench

        inner_radius_2 (float): Inner radius of second end of wrench

        height_2 (float): Height of second end of wrench

        ngeoms (int): Number of box geoms used to approximate the ends of the wrench. Use
            more geoms to make the approximation better.
    """

    def __init__(
        self,
        name,
        handle_size=(0.08, 0.01, 0.005),
        outer_radius_1=0.0425,
        inner_radius_1=0.03,
        height_1=0.05,
        outer_radius_2=0.0425,
        inner_radius_2=0.03,
        height_2=0.05,
        ngeoms=8,
        # rgba=None,
        density=1000.,
        solref=(0.02, 1.),
        solimp=(0.9, 0.95, 0.001),
        friction=None,
    ):
        # Object properties
        self.handle_size = tuple(handle_size)
        self.outer_radii = (outer_radius_1, outer_radius_2)
        self.inner_radii = (inner_radius_1, inner_radius_2)
        self.heights = (height_1, height_2)
        self.ngeoms = ngeoms

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        wrench_mat = CustomMaterial(
            texture="SteelScratched",
            tex_name="steel",
            mat_name="steel_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # Create objects
        objects = []

        # each end of the wrench is modeled by a hollow cylinder
        for i in range(2):
            objects.append(HollowCylinderObject(
                name=f"hole{i + 1}",
                outer_radius=self.outer_radii[i],
                inner_radius=self.inner_radii[i],
                height=self.heights[i],
                ngeoms=self.ngeoms,
                rgba=None,
                material=wrench_mat,
                density=density,
                solref=solref,
                solimp=solimp,
                friction=friction,
                make_half=False,
            ))

        # also add center box geom for handle
        objects.append(BoxObject(
            name="handle",
            size=handle_size,
            rgba=None,
            material=wrench_mat,
            density=density,
            solref=solref,
            solimp=solimp,
            friction=friction,
        ))

        # Define positions (top-level body is centered at handle)
        hole_1_box_geom_height = 2. * objects[0].unit_box_height
        hole_2_box_geom_height = 2. * objects[1].unit_box_height
        positions = [
            # this computation ensures no gaps between the center bar geom and the two wrench holes at the end
            np.array([-handle_size[0] - self.outer_radii[0] + hole_1_box_geom_height, 0, 0]),
            np.array([handle_size[0] + self.outer_radii[1] - hole_2_box_geom_height, 0, 0]),
            np.zeros(3),
        ]
        quats = [None, None, None]
        parents = [None, None, None]

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=positions,
            object_quats=quats,
            object_parents=parents,
            joints=[dict(type="free", damping="0.0005")], # be consistent with round-nut.xml
        )
