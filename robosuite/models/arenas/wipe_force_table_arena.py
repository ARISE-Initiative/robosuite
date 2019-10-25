import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from collections import OrderedDict
from robosuite.models.objects import CylinderObject, PlateWithHoleObject, BoxObject

import xml.etree.ElementTree as ET

import random
import itertools

from pyquaternion import Quaternion

class WipeForceTableArena(Arena):
    """Workspace that contains an empty table with tactile sensors on its surface."""

    def __init__(
        self, 
        table_full_size=(0.8, 0.8, 0.8), 
        table_friction=(0.01, 0.005, 0.0001), 
        num_squares=(10,10),
        prob_sensor=1.0,
        rotation_x=0,
        rotation_y=0,
        draw_line=True,
        num_sensors=10,
        table_friction_std=0,
        line_width=0.02,
        two_clusters=False
    ):
        """
        Args:
            table_full_size: full dimensions of the table
            friction: friction parameters of the table
            num_squares: number of squares in each dimension of the table top
        """
        super().__init__(xml_path_completion("arenas/wipe_force_table_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_friction_std = table_friction_std
        self.line_width = line_width

        self.num_squares = np.array(num_squares)

        self.sensor_names = []
        self.sensor_site_names = {}

        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")

        self.coverage_factor = 1.0 #How much of the table surface we cover

        self.prob_sensor = prob_sensor

        self.rotation_x = rotation_x
        self.rotation_y = rotation_y

        self.draw_line = draw_line
        self.num_sensors = num_sensors
        self.two_clusters = two_clusters

        self.configure_location()

    def configure_location(self):
        self.bottom_pos = np.array([0, 0, 0])
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.center_pos = self.bottom_pos + np.array([0, 0, self.table_half_size[2]])
        self.table_body.set("pos", array_to_string(self.center_pos))

        qx= Quaternion(axis=(1.0, 0.0, 0.0), radians=self.rotation_x)
        qy= Quaternion(axis=(0.0, 1.0, 0.0), radians=self.rotation_y)
        qt = qy*qx

        friction= max(0.001, np.random.normal(self.table_friction[0], self.table_friction_std))
        print("New table friction:" + str(friction))

        self.table_body.set("quat", array_to_string(qt.elements))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        #Compute size of the squares
        self.square_full_size = np.divide(self.table_full_size[0:2]*self.coverage_factor, self.num_squares)
        self.square_half_size = self.square_full_size/2.0

        self.peg_size = 0.01

        if self.draw_line: 
            self.squares = OrderedDict()
            self.mujoco_objects = OrderedDict()

            table_subtree = self.worldbody.find(".//body[@name='{}']".format("table"))

            # Sites on additional bodies attached to the table
            #squares_separation = 0.0005
            squares_separation = 0.0
            squares_height = 0.1
            table_cte_size_x = 0.8
            table_cte_size_y = 1.0
            
            square2 = BoxObject(size=[table_cte_size_x/2, table_cte_size_y/2,  squares_height/2 - 0.001],
                               rgba=[0, 0,1, 1],
                               density=1,
                               friction=friction
                               )
            square_name2 = 'table_0_0'
            self.squares[square_name2] = square2
            collision_c = square2.get_collision(name=square_name2, site=True)

            #Align the constant sized table to the bottom of the table
            collision_c.set("pos", array_to_string([table_cte_size_x/2 - self.table_half_size[0] - 0.18,
                0,
                self.table_half_size[2]+squares_height/2]))
            table_subtree.append(collision_c)

            pos = np.array((np.random.uniform(-self.table_half_size[0], self.table_half_size[0]),
                np.random.uniform(-self.table_half_size[1], self.table_half_size[1])))
            direction =  np.random.uniform(-np.pi, np.pi)



            if not self.two_clusters:
                for i in range(self.num_sensors):
                    square2 = CylinderObject(size=[self.line_width, 0.001],
                                       rgba=[0, 1,0, 0],
                                       density=1,
                                       friction=friction
                                       )
                    square_name2 = 'contact_'+str(i)
                    self.squares[square_name2] = square2
                    collision_c = square2.get_collision(name=square_name2, site=True)
                    collision_c.set("pos", array_to_string([pos[0],pos[1],self.table_half_size[2]+squares_height]))
                    #collision_c.find("site").set("pos", array_to_string([0,0,0]))
                    collision_c.find("site").set("pos", array_to_string([pos[0],pos[1],self.table_half_size[2]+squares_height+0.005]))
                    collision_c.find("site").set("size", array_to_string([self.line_width, 0.001]))
                    collision_c.find("geom").set("contype", "0")
                    collision_c.find("geom").set("conaffinity", "0")      
                    collision_c.find("site").set("rgba", array_to_string([0,1,0, 1]))
                    #table_subtree.append(collision_c)
                    table_subtree.append(collision_c.find("site"))

                    sensor_name = square_name2 +"_sensor"
                    sensor_site_name = square_name2# +"_site"
                    self.sensor_names += [sensor_name]
                    self.sensor_site_names[sensor_name] = sensor_site_name
                    #ET.SubElement(self.sensor, "touch", attrib={"name" : sensor_name, "site" : sensor_site_name})

                    if np.random.uniform(0,1) > 0.7:
                        direction += np.random.normal(0, 0.5)

                    posnew0 = pos[0] + 0.005*np.sin(direction)
                    posnew1 = pos[1] + 0.005*np.cos(direction)

                    while abs(posnew0) >= self.table_half_size[0] or abs(posnew1) >= self.table_half_size[1]:
                        direction += np.random.normal(0, 0.5)
                        posnew0 = pos[0] + 0.005*np.sin(direction)
                        posnew1 = pos[1] + 0.005*np.cos(direction)

                    pos[0] = posnew0
                    pos[1] = posnew1
            else:
                half_num_sensors = int(np.floor(self.num_sensors/2))
                last_i = 0
                for i in range(half_num_sensors):
                    square2 = CylinderObject(size=[self.line_width, 0.001],
                                       rgba=[0, 1,0, 0],
                                       density=1,
                                       friction=friction
                                       )
                    square_name2 = 'contact_'+str(i)
                    self.squares[square_name2] = square2
                    collision_c = square2.get_collision(name=square_name2, site=True)
                    collision_c.set("pos", array_to_string([pos[0],pos[1],self.table_half_size[2]+squares_height+0.01]))
                    #collision_c.find("site").set("pos", array_to_string([0,0,0]))
                    collision_c.find("site").set("pos", array_to_string([pos[0],pos[1],self.table_half_size[2]+squares_height+0.005]))
                    collision_c.find("site").set("size", array_to_string([self.line_width, 0.001]))
                    collision_c.find("geom").set("contype", "0")
                    collision_c.find("geom").set("conaffinity", "0")      
                    collision_c.find("site").set("rgba", array_to_string([0,1,0, 1]))
                    #table_subtree.append(collision_c)
                    table_subtree.append(collision_c.find("site"))

                    sensor_name = square_name2 +"_sensor"
                    sensor_site_name = square_name2# +"_site"
                    self.sensor_names += [sensor_name]
                    self.sensor_site_names[sensor_name] = sensor_site_name
                    #ET.SubElement(self.sensor, "touch", attrib={"name" : sensor_name, "site" : sensor_site_name})

                    if np.random.uniform(0,1) > 0.7:
                        direction += np.random.normal(0, 0.5)

                    posnew0 = pos[0] + 0.005*np.sin(direction)
                    posnew1 = pos[1] + 0.005*np.cos(direction)

                    while abs(posnew0) >= self.table_half_size[0] or abs(posnew1) >= self.table_half_size[1]:
                        direction += np.random.normal(0, 0.5)
                        posnew0 = pos[0] + 0.005*np.sin(direction)
                        posnew1 = pos[1] + 0.005*np.cos(direction)

                    pos[0] = posnew0
                    pos[1] = posnew1

                    last_i = i


                last_i+=1
                pos = np.array((np.random.uniform(-self.table_half_size[0], self.table_half_size[0]),
                    np.random.uniform(-self.table_half_size[1], self.table_half_size[1])))
                direction =  np.random.uniform(-np.pi, np.pi)

                for i in range(half_num_sensors):
                    square2 = CylinderObject(size=[self.line_width, 0.001],
                                       rgba=[0, 1,0, 0],
                                       density=1,
                                       friction=friction
                                       )
                    square_name2 = 'contact_'+str(last_i +i)
                    self.squares[square_name2] = square2
                    collision_c = square2.get_collision(name=square_name2, site=True)
                    collision_c.set("pos", array_to_string([pos[0],pos[1],self.table_half_size[2]+squares_height+0.01]))
                    #collision_c.find("site").set("pos", array_to_string([0,0,0]))
                    collision_c.find("site").set("pos", array_to_string([pos[0],pos[1],self.table_half_size[2]+squares_height+0.005]))
                    collision_c.find("site").set("size", array_to_string([self.line_width, 0.001]))
                    collision_c.find("geom").set("contype", "0")
                    collision_c.find("geom").set("conaffinity", "0")      
                    collision_c.find("site").set("rgba", array_to_string([0,1,0, 1]))
                    #table_subtree.append(collision_c)
                    table_subtree.append(collision_c.find("site"))

                    sensor_name = square_name2 +"_sensor"
                    sensor_site_name = square_name2# +"_site"
                    self.sensor_names += [sensor_name]
                    self.sensor_site_names[sensor_name] = sensor_site_name
                    #ET.SubElement(self.sensor, "touch", attrib={"name" : sensor_name, "site" : sensor_site_name})

                    if np.random.uniform(0,1) > 0.7:
                        direction += np.random.normal(0, 0.5)

                    posnew0 = pos[0] + 0.005*np.sin(direction)
                    posnew1 = pos[1] + 0.005*np.cos(direction)

                    while abs(posnew0) >= self.table_half_size[0] or abs(posnew1) >= self.table_half_size[1]:
                        direction += np.random.normal(0, 0.5)
                        posnew0 = pos[0] + 0.005*np.sin(direction)
                        posnew1 = pos[1] + 0.005*np.cos(direction)

                    pos[0] = posnew0
                    pos[1] = posnew1

        else:

            self.squares = OrderedDict()

            indices = [a for a in itertools.product(range(self.num_squares[0]), range(self.num_squares[1]))]
            picked_indices = random.sample(indices, int(np.ceil(self.prob_sensor*len(indices))))

            for i in range(self.num_squares[0]):
                for j in range(self.num_squares[1]):

                    self.mujoco_objects = OrderedDict()

                    table_subtree = self.worldbody.find(".//body[@name='{}']".format("table"))

                    # Sites directly attached to the table
                    # square = BoxObject(size=[self.square_half_size[0]-0.0005, self.square_half_size[1]-0.0005, 0.001],
                    #                    rgba=[0, 0, 1, 1],
                    #                    density=500,
                    #                    friction=0.05)
                    # square_name = 'contact_'+str(i) + "_" + str(j)
                    # self.squares[square_name] = square
                    # collision_b = square.get_collision(name=square_name, site=True)
                    # collision_b.set("pos", array_to_string([
                    #     self.table_half_size[0]-i*self.square_full_size[0]-0.5*self.square_full_size[0], 
                    #     self.table_half_size[1]-j*self.square_full_size[1]-0.5*self.square_full_size[1], 
                    #     self.table_half_size[2]+0.05]))  
                    # collision_b.find("site").set("pos", array_to_string([0,0,0.002])) 
                    # collision_b.find("site").set("pos", array_to_string([
                    #     self.table_half_size[0]-i*self.square_full_size[0]-0.5*self.square_full_size[0], 
                    #     self.table_half_size[1]-j*self.square_full_size[1]-0.5*self.square_full_size[1], 
                    #     self.table_half_size[2]+0.005])) 
                    # table_subtree.append(collision_b.find("site"))
                    # sensor_name = square_name +"_sensor"
                    # self.sensor_names += [sensor_name]
                    # ET.SubElement(self.sensor, "force", attrib={"name" : sensor_name, "site" : square_name + "_site"})
                    #print(ET.tostring(collision_b.find("site")))

                    # Sites on additional bodies attached to the table
                    #squares_separation = 0.0005
                    squares_separation = 0.0
                    squares_height = 0.1
                    square2 = BoxObject(size=[self.square_half_size[0]-0.0005, self.square_half_size[1]-squares_separation, squares_height/2 - 0.001],
                                       rgba=[0, 1, 0, 1],
                                       density=1,
                                       friction=0.0001)
                    square_name2 = 'contact_'+str(i) + "_" + str(j)
                    self.squares[square_name2] = square2
                    collision_c = square2.get_collision(name=square_name2, site=True)
                    collision_c.set("pos", array_to_string([
                        self.table_half_size[0]-i*self.square_full_size[0]-0.5*self.square_full_size[0], 
                        self.table_half_size[1]-j*self.square_full_size[1]-0.5*self.square_full_size[1], 
                        self.table_half_size[2]+squares_height/2 - 0.001]))
                    collision_c.find("site").set("pos", array_to_string([0,0,squares_height/2]))
                    collision_c.find("site").set("size", array_to_string([self.square_half_size[0]-squares_separation, self.square_half_size[1]-squares_separation,0.002]))
                    place_this_sensor = (i,j) in picked_indices
                    if i < self.num_squares[0] - 1 and place_this_sensor: 
                        collision_c.find("site").set("rgba", array_to_string([0,1,0, 1]))
                    else:
                        collision_c.find("site").set("rgba", array_to_string([0,0,1, 1]))
                    table_subtree.append(collision_c)
                    #Except the closest row to the robot
                    if i < self.num_squares[0] - 1 and place_this_sensor:

                        sensor_name = square_name2 +"_sensor"
                        sensor_site_name = square_name2# +"_site"
                        self.sensor_names += [sensor_name]
                        self.sensor_site_names[sensor_name] = sensor_site_name
                        ET.SubElement(self.sensor, "force", attrib={"name" : sensor_name, "site" : sensor_site_name})      

            #Add upper border to the table
            table_subtree = self.worldbody.find(".//body[@name='{}']".format("table"))
            squares_height = 0.1
            border_half_size = 0.08
            square2 = BoxObject(size=[border_half_size, self.table_half_size[1] + 2*border_half_size, squares_height/2 - 0.001],
                               rgba=[0, 1, 0, 1],
                               density=1,
                               friction=0.05)
            square_name2 = 'border_upper'
            self.squares[square_name2] = square2
            collision_c = square2.get_collision(name=square_name2, site=True)
            collision_c.set("pos", array_to_string([self.table_half_size[0]+border_half_size, 0, self.table_half_size[2]+squares_height/2 - 0.001]))
            collision_c.find("site").set("pos", array_to_string([0,0,squares_height/2]))
            collision_c.find("site").set("size", array_to_string([border_half_size, self.table_half_size[1]+ 2*border_half_size, 0.002]))
            collision_c.find("site").set("rgba", array_to_string([0,0,1, 1]))
            table_subtree.append(collision_c)

            

            #Add left border to the table
            square2 = BoxObject(size=[self.table_half_size[0], border_half_size, squares_height/2 - 0.001],
                               rgba=[0, 1, 0, 1],
                               density=1,
                               friction=0.05)
            square_name2 = 'border_left'
            self.squares[square_name2] = square2
            collision_c = square2.get_collision(name=square_name2, site=True)
            collision_c.set("pos", array_to_string([0, self.table_half_size[1]+border_half_size, self.table_half_size[2]+squares_height/2 - 0.001]))
            collision_c.find("site").set("pos", array_to_string([0,0,squares_height/2]))
            collision_c.find("site").set("size", array_to_string([self.table_half_size[0], border_half_size, 0.002]))
            collision_c.find("site").set("rgba", array_to_string([0,0,1, 1]))
            table_subtree.append(collision_c)

            #Add right border to the table
            square2 = BoxObject(size=[self.table_half_size[0], border_half_size, squares_height/2 - 0.001],
                               rgba=[0, 1, 0, 1],
                               density=1,
                               friction=0.05)
            square_name2 = 'border_right'
            self.squares[square_name2] = square2
            collision_c = square2.get_collision(name=square_name2, site=True)
            collision_c.set("pos", array_to_string([0, -self.table_half_size[1]-border_half_size, self.table_half_size[2]+squares_height/2 - 0.001]))
            collision_c.find("site").set("pos", array_to_string([0,0,squares_height/2]))
            collision_c.find("site").set("size", array_to_string([self.table_half_size[0], border_half_size, 0.002]))
            collision_c.find("site").set("rgba", array_to_string([0,0,1, 1]))
            table_subtree.append(collision_c)

            #Add lower border to the table
            low_border_half_size = 0.06
            square2 = BoxObject(size=[low_border_half_size, self.table_half_size[1] + 2*border_half_size, squares_height/2 - 0.001],
                               rgba=[0, 1, 0, 1],
                               density=1,
                               friction=0.05)
            square_name2 = 'border_lower'
            self.squares[square_name2] = square2
            collision_c = square2.get_collision(name=square_name2, site=True)
            collision_c.set("pos", array_to_string([-self.table_half_size[0]-low_border_half_size, 0, self.table_half_size[2]+squares_height/2 - 0.001]))
            collision_c.find("site").set("pos", array_to_string([0,0,squares_height/2]))
            collision_c.find("site").set("size", array_to_string([low_border_half_size, self.table_half_size[1]+ 2*border_half_size, 0.002]))
            collision_c.find("site").set("rgba", array_to_string([0,0,1, 1]))
            table_subtree.append(collision_c)



        self.table_top.set(
            "pos", array_to_string(np.array([0, 0, self.table_half_size[2]]))
        )

    @property
    def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height

