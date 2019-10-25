import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from collections import OrderedDict
from robosuite.models.objects import CylinderObject, PlateWithHoleObject, BoxObject

import xml.etree.ElementTree as ET

class TactileTableArena(Arena):
    """Workspace that contains an empty table with tactile sensors on its surface."""

    def __init__(
        self, table_full_size=(0.8, 0.8, 0.8), 
        table_friction=(1, 0.005, 0.0001), 
        num_squares=(10,10)
    ):
        """
        Args:
            table_full_size: full dimensions of the table
            friction: friction parameters of the table
            num_squares: number of squares in each dimension of the table top
        """
        super().__init__(xml_path_completion("arenas/tactile_table_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.num_squares = np.array(num_squares)

        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")

        self.configure_location()

    def configure_location(self):
        self.bottom_pos = np.array([0, 0, 0])
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.center_pos = self.bottom_pos + np.array([0, 0, self.table_half_size[2]])
        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        #Compute size of the squares
        self.square_full_size = np.divide(self.table_full_size[0:2], self.num_squares)
        self.square_half_size = np.divide(self.table_half_size[0:2], self.num_squares)

        self.squares = OrderedDict()

        for i in range(self.num_squares[0]):
            for j in range(self.num_squares[1]):
                self.mujoco_objects = OrderedDict()


                square = BoxObject(size=[self.square_half_size[0]-0.0005, self.square_half_size[1]-0.0005, 0.001],
                                   rgba=[0, 0, 1, 1],
                                   density=500,
                                   friction=0.05)

                square_name = 'contact_'+str(i) + "_" + str(j)
                self.squares[square_name] = square

                table_subtree = self.worldbody.find(".//body[@name='{}']".format("table"))
                
                collision_b = square.get_collision(name=square_name, site=True)
                collision_b.set("pos", array_to_string([
                    self.table_half_size[0]-i*self.square_full_size[0]-0.5*self.square_full_size[0], 
                    self.table_half_size[1]-j*self.square_full_size[1]-0.5*self.square_full_size[1], 
                    self.table_half_size[2]+0.05]))  
                collision_b.find("site").set("pos", array_to_string([0,0,0.002])) 

                collision_b.find("site").set("pos", array_to_string([
                    self.table_half_size[0]-i*self.square_full_size[0]-0.5*self.square_full_size[0], 
                    self.table_half_size[1]-j*self.square_full_size[1]-0.5*self.square_full_size[1], 
                    self.table_half_size[2]+0.005])) 

                table_subtree.append(collision_b.find("site"))

                #print(ET.tostring(collision_b.find("site")))

                # square2 = BoxObject(size=[self.square_half_size[0]-0.0005, self.square_half_size[1]-0.0005, 0.03/2 - 0.001],
                #                    rgba=[1, 0, 0, 1],
                #                    density=500,
                #                    friction=0.05)

                # collision_c = square2.get_collision(name=square_name, site=True)
                # #print(ET.tostring(collision_c))
                # #print(50*'dd')
                # collision_c.set("pos", array_to_string([
                #     self.table_half_size[0]-i*self.square_full_size[0]-0.5*self.square_full_size[0], 
                #     self.table_half_size[1]-j*self.square_full_size[1]-0.5*self.square_full_size[1], 
                #     self.table_half_size[2]+0.03/2 - 0.001]))
                # collision_c.find("site").set("pos", array_to_string([0,0,0.03/2]))
                # collision_c.find("site").set("size", array_to_string([self.square_half_size[0]-0.0005, self.square_half_size[1]-0.0005,0.002]))
                # collision_c.find("site").set("rgba", array_to_string([0,0,1, 1]))
                # table_subtree.append(collision_c)

                #print(ET.tostring(collision_c))

                ET.SubElement(self.sensor, "touch", attrib={"name" : square_name +"_sensor", "site" : square_name})# + "_site"})

        #print(self.get_xml())
        #exit()      

        self.table_top.set(
            "pos", array_to_string(np.array([0, 0, self.table_half_size[2]]))
        )

    @property
    def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height
