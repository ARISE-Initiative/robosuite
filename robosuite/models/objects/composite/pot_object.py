import numpy as np
from robosuite.models.objects import MujocoXMLObject, CompositeObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string, find_elements, CustomMaterial, add_to_dict, RED, GREEN, BLUE
from robosuite.models.objects import BoxObject
import robosuite.utils.transform_utils as T



class PotObject(CompositeObject):
    def __init__(
            self,
            name,
            tunnel_size=(0.1, 0.015, 0.06),
            density=1000,
            use_texture=True):

        self._name = name
        self.length = tunnel_size[1]
        self.height = tunnel_size[2]

        self.use_texture = use_texture

        base_args = {
            "total_size": self.length / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        site_attrs = []
        obj_args = {}

        # Long sides
        flat_size = [tunnel_size[0] - 0.04, tunnel_size[1], 0.005]
        side_size = [0.02, tunnel_size[1], tunnel_size[2] + 0.005]

        r = np.pi / 2

        geom_mat = "steel_scratched_mat"
        pot_width = 0.0
        pot_length = 0.06

        edge_width = 0.007
        geom_frictions = (0.005, 0.005, 0.0001)

        solref = (0.02, 1.)
        
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0., 0., 0.0025),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([pot_length - 0.01 - 0.007, pot_length - 0.007, 0.005]),
            geom_names=f"body_0",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=(0.0005, 0.005, 0.0001),
            # geom_frictions=(0.0, 0.0, 0.0),
            solref=solref,
        )

        pot_height = 0.035
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(pot_width, -pot_length, pot_height - 0.0025),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([pot_length + 0.007 - 0.01, edge_width, pot_height]),
            geom_names=f"body_1",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,
            density=density)

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(pot_width, pot_length, pot_height - 0.0025),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([pot_length + 0.007 - 0.01, edge_width, pot_height]),
            geom_names=f"body_2",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,                                    
            density=density)

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(pot_length - 0.01, pot_width, pot_height - 0.0025),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([0.007, pot_length - 0.007, pot_height]),
            geom_names=f"body_3",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,                                    
            density=density)

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(-pot_length + 0.01, pot_width, pot_height - 0.0025),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([0.007, pot_length - 0.007, pot_height]),
            geom_names=f"body_4",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,                                    
            density=density)

        handle_radius = 0.01
        handle_width = 0.055
        handle_length = edge_width * 2
        handle_friction = 1.0

        for (direction, y) in zip(['left', 'right'], [pot_length , -pot_length]):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(0.0, y, 2 * pot_height + 0.007),
                geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
                geom_sizes=np.array([handle_width / 2, handle_length / 2, handle_radius]),
                geom_names=f"handle_{direction}_1",
                geom_rgbas=None,
                geom_materials=geom_mat,
                # geom_frictions=(handle_friction, 0.005, 0.0001),
                geom_frictions=geom_frictions,
                solref=solref,                                
                density=density)
        
        bottom_site = self.get_site_attrib_template()
        top_site = self.get_site_attrib_template()
        horizontal_site = self.get_site_attrib_template()

        bottom_site.update({
            "name": "bottom",
            "pos": array_to_string(np.array([0., 0., -2 * self.height])),
            "size": "0.005",
            "rgba": "0 0 0 0"
        })

        top_site.update({
            "name": "top",
            "pos": array_to_string(np.array([0., 0., 2 * self.height])),
            "size": "0.005",
            "rgba": "0 0 0 0"
        })

        bottom_site.update({
            "name": "bottom",
            "pos": array_to_string(np.array([0., 0., -2 * self.height])),
            "size": "0.005",
            "rgba": "0 0 0 0"
        })

        obj_args.update(base_args)

        obj_args["sites"] = site_attrs
        obj_args["joints"] = [{"type": "free", "damping":"0.0005"}]

        super().__init__(**obj_args)
        

        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        steel_scratched_material = CustomMaterial(
            texture="SteelScratched",
            tex_name="steel_scratched_tex",
            mat_name="steel_scratched_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
 
        self.append_material(steel_scratched_material)

    @property
    def bottom_offset(self):
        return np.array([0, 0, -2 * self.height])

    @property
    def top_offset(self):
        return np.array([0, 0, 2 * self.height])
        
    @property
    def horizontal_radius(self):
        return self.length * np.sqrt(2)
