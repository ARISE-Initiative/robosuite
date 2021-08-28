from robosuite.models.objects import CompositeBodyObject, BoxObject, CylinderObject
import numpy as np

from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.mjcf_utils import RED, BLUE, CustomMaterial


class HingedBoxObject(CompositeBodyObject):
    """
    An example object that demonstrates the CompositeBodyObject functionality. This object consists of two cube bodies
    joined together by a hinge joint.

    Args:
        name (str): Name of this object

        box1_size (3-array): (L, W, H) half-sizes for the first box

        box2_size (3-array): (L, W, H) half-sizes for the second box

        use_texture (bool): set True if using wood textures for the blocks
    """

    def __init__(
        self,
        name,
        box1_size=(0.025, 0.025, 0.025),
        box2_size=(0.025, 0.025, 0.0125),
        use_texture=True,
    ):
        # Set box sizes
        self.box1_size = np.array(box1_size)
        self.box2_size = np.array(box2_size)

        # Set texture attributes
        self.use_texture = use_texture
        self.box1_material = None
        self.box2_material = None
        self.box1_rgba = RED
        self.box2_rgba = BLUE

        # Define materials we want to use for this object
        if self.use_texture:
            # Remove RGBAs
            self.box1_rgba = None
            self.box2_rgba = None

            # Set materials for each box
            tex_attrib = {
                "type": "cube",
            }
            mat_attrib = {
                "texrepeat": "3 3",
                "specular": "0.4",
                "shininess": "0.1",
            }
            self.box1_material = CustomMaterial(
                texture="WoodRed",
                tex_name="box1_tex",
                mat_name="box1_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            self.box2_material = CustomMaterial(
                texture="WoodBlue",
                tex_name="box2_tex",
                mat_name="box2_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )

        # Create objects
        objects = []
        for i, (size, mat, rgba) in enumerate(
            zip(
                (self.box1_size, self.box2_size),
                (self.box1_material, self.box2_material),
                (self.box1_rgba, self.box2_rgba),
            )
        ):
            objects.append(
                BoxObject(
                    name=f"box{i + 1}",
                    size=size,
                    rgba=rgba,
                    material=mat,
                )
            )

        # Also add hinge for visualization
        objects.append(
            CylinderObject(
                name="hinge",
                size=np.array(
                    [
                        min(self.box1_size[2], self.box2_size[2]) / 5.0,
                        min(self.box1_size[0], self.box2_size[0]),
                    ]
                ),
                rgba=[0.5, 0.5, 0, 1],
                obj_type="visual",
            )
        )

        # Define hinge joint
        rel_hinge_pos = [
            self.box2_size[0],
            0,
            -self.box2_size[2],
        ]  # want offset in all except y-axis
        hinge_joint = {
            "name": "box_hinge",
            "type": "hinge",
            "axis": "0 1 0",  # y-axis hinge
            "pos": array_to_string(rel_hinge_pos),
            "stiffness": "0.0001",
            "limited": "true",
            "range": "0 1.57",
        }

        # Define positions -- second box should lie on top of first box with edge aligned at hinge joint
        # Hinge visualizer should be aligned at hinge joint location
        positions = [
            np.zeros(3),  # First box is centered at top-level body anyways
            np.array(
                [-(self.box2_size[0] - self.box1_size[0]), 0, self.box1_size[2] + self.box2_size[2]]
            ),
            np.array(rel_hinge_pos),
        ]

        quats = [
            None,  # Default quaternion for box 1
            None,  # Default quaternion for box 2
            [0.707, 0.707, 0, 0],  # Rotated 90 deg about x-axis
        ]

        # Define parents -- which body each is aligned to
        parents = [
            None,  # box 1 attached to top-level body
            objects[0].root_body,  # box 2 attached to box 1
            objects[1].root_body,  # hinge attached to box 2
        ]

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=positions,
            object_quats=quats,
            object_parents=parents,
            body_joints={objects[1].root_body: [hinge_joint]},
        )
