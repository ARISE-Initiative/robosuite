import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from collections import OrderedDict
from robosuite.models.objects import CylinderObject, PlateWithHoleObject, BoxObject, BallObject

import xml.etree.ElementTree as ET

import imageio


class HeightTableArena(Arena):
    """Workspace that contains an empty table."""

    def __init__(
        self, table_height_full_size=(0.8, 0.8, 0.8, 0.8), table_friction=(1, 0.005, 0.0001)
    ):
        """
        Args:
            table_full_size: full dimensions of the table height: x, y, height scale over the surface, box depth from the min
            friction: friction parameters of the table
        """
        super().__init__(xml_path_completion("arenas/height_table_arena.xml"))

        self.table_height_full_size = np.array(table_height_full_size)
        self.table_half_size = np.array([
                self.table_height_full_size[0],
                self.table_height_full_size[1],
                self.table_height_full_size[3]]) / 2

        self.table_friction = table_friction

        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.table_body = self.worldbody.find("./geom[@name='table_hf_geom']")
        self.table_top_abs = self.worldbody.find("./geom[@name='table_hf_geom']")
        self.table_asset = self.asset.find("./hfield[@name='table_hf']")

        self.sensor_sites = {}

        self.configure_location()

    def configure_location(self):
        self.bottom_pos = np.array([0, 0, 0])
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.center_pos = self.bottom_pos + np.array([0, 0, 2*self.table_half_size[2]])
        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_asset.set("size", array_to_string(self.table_height_full_size))
        self.table_body.set("friction", array_to_string(self.table_friction))
            
        height_img = imageio.imread(xml_path_completion(self.table_asset.get("file")), as_gray=True)

        total_pixels = height_img.shape[0] * height_img.shape[1]

        #TODO: the number of sensors affects the performance. Do not increse over 500
        #800 sensors -> mujoco steps up to 0.25sec per policy step!
        num_sensors = min(500, total_pixels)

        sample_rate = np.floor(total_pixels / num_sensors)

        #Random Sampling:
        used_pairs = []
        sensor_counter = 0
        while sensor_counter < num_sensors:
            i = np.random.randint(0, high=height_img.shape[0])
            j = np.random.randint(0, high=height_img.shape[1])

            if (i,j) not in used_pairs: # We could add additional constraints like that sensors should be appart from each other

                self.mujoco_objects = OrderedDict()
                square = BallObject(size=[0.01],
                                   rgba=[0, 0, 1, 1],
                                   density=500,
                                   friction=0.05)
                square_name = 'contact_'+str(i)+"_"+str(j)
                collision_b = square.get_collision(name=square_name, site=True)

                x_pos = 0
                if height_img.shape[0]%2==0:
                    x_pos = -(np.floor(height_img.shape[0]/2) - 0.5 - i)*2*self.table_height_full_size[0]/height_img.shape[0]
                else:
                    x_pos = (i-np.floor(height_img.shape[0]/2))*2*self.table_height_full_size[0]/height_img.shape[0]

                y_pos = 0
                if height_img.shape[1]%2==0:
                    y_pos = -(np.floor(height_img.shape[1]/2)-0.5-j)*2*self.table_height_full_size[1]/height_img.shape[1]
                else:
                    y_pos = (j-np.floor(height_img.shape[1]/2))*2*self.table_height_full_size[1]/height_img.shape[1]

                #Compute mean of pixel value and neighbors -> better for low dim images
                acc = 0
                mean_ctr = 1
                pixel_value = height_img[i,j]

                # DO NOT DELETE! The following code helps for low dimensional pictures because the interpolation "buries" some sensors
                # if i+1< height_img.shape[0]:
                #     if height_img[i+1,j] > pixel_value:
                #         acc += height_img[i+1,j]
                #         mean_ctr += 1
                # if i-1<=0:
                #     if height_img[i-1,j] > pixel_value:
                #         acc += height_img[i-1,j]
                #         mean_ctr += 1
                # if j+1< height_img.shape[0]:
                #     if height_img[i,j+1] > pixel_value:
                #         acc += height_img[i,j+1]
                #         mean_ctr += 1
                # if j-1<=0:
                #     if height_img[i,j-1] > pixel_value:
                #         acc += height_img[i,j-1]
                #         mean_ctr += 1

                acc += height_img[i,j]
                z_pos = self.table_height_full_size[3] + self.table_height_full_size[2]*(acc/float(mean_ctr))/np.max(height_img)+0.005

                position_sensor = [y_pos, x_pos,z_pos]
                position_sensor[1] = -position_sensor[1]

                collision_b.find("site").set("pos", array_to_string(position_sensor)) 
                self.worldbody.append(collision_b.find("site"))
                self.sensor_sites[sensor_counter] = position_sensor
                
                # Add this pixel to the list of used pixels
                used_pairs += [(i,j)]

                # Add sensor linked to the sensor site
                ET.SubElement(self.sensor, "touch", attrib={"name" : square_name +"_sensor", "site" : square_name})# + "_site"})
                sensor_counter+=1

