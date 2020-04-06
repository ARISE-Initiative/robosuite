import random
import numpy as np
import xml.etree.ElementTree as ET

from robosuite.models.objects import MujocoGeneratedObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site
from robosuite.utils.mjcf_utils import RED, GREEN, BLUE


class PotWithHandlesObject(MujocoGeneratedObject):
    """
    Generates the Pot object with side handles (used in BaxterLift)
    """

    def __init__(
        self,
        body_half_size=None,
        handle_radius=0.01,
        handle_length=0.09,
        handle_width=0.09,
        rgba_body=None,
        rgba_handle_1=None,
        rgba_handle_2=None,
        solid_handle=False,
        thickness=0.025,  # For body
    ):
        super().__init__()
        if body_half_size:
            self.body_half_size = body_half_size
        else:
            self.body_half_size = np.array([0.07, 0.07, 0.07])
        self.thickness = thickness
        self.handle_radius = handle_radius
        self.handle_length = handle_length
        self.handle_width = handle_width
        if rgba_body:
            self.rgba_body = np.array(rgba_body)
        else:
            self.rgba_body = RED
        if rgba_handle_1:
            self.rgba_handle_1 = np.array(rgba_handle_1)
        else:
            self.rgba_handle_1 = GREEN
        if rgba_handle_2:
            self.rgba_handle_2 = np.array(rgba_handle_2)
        else:
            self.rgba_handle_2 = BLUE
        self.solid_handle = solid_handle

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.body_half_size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.body_half_size[2]])

    def get_horizontal_radius(self):
        return np.sqrt(2) * (max(self.body_half_size) + self.handle_length)

    @property
    def handle_distance(self):
        return self.body_half_size[1] * 2 + self.handle_length * 2

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)

        for geom in five_sided_box(
            self.body_half_size, self.rgba_body, 1, self.thickness
        ):
            main_body.append(geom)
        handle_z = self.body_half_size[2] - self.handle_radius
        handle_1_center = [0, self.body_half_size[1] + self.handle_length, handle_z]
        handle_2_center = [
            0,
            -1 * (self.body_half_size[1] + self.handle_length),
            handle_z,
        ]
        # the bar on handle horizontal to body
        main_bar_size = [
            self.handle_width / 2 + self.handle_radius,
            self.handle_radius,
            self.handle_radius,
        ]
        side_bar_size = [self.handle_radius, self.handle_length / 2, self.handle_radius]
        handle_1 = new_body(name="handle_1")
        if self.solid_handle:
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1",
                    pos=[0, self.body_half_size[1] + self.handle_length / 2, handle_z],
                    size=[
                        self.handle_width / 2,
                        self.handle_length / 2,
                        self.handle_radius,
                    ],
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
        else:
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_c",
                    pos=handle_1_center,
                    size=main_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_+",  # + for positive x
                    pos=[
                        self.handle_width / 2,
                        self.body_half_size[1] + self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_-",
                    pos=[
                        -self.handle_width / 2,
                        self.body_half_size[1] + self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )

        handle_2 = new_body(name="handle_2")
        if self.solid_handle:
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2",
                    pos=[0, -self.body_half_size[1] - self.handle_length / 2, handle_z],
                    size=[
                        self.handle_width / 2,
                        self.handle_length / 2,
                        self.handle_radius,
                    ],
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
        else:
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_c",
                    pos=handle_2_center,
                    size=main_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_+",  # + for positive x
                    pos=[
                        self.handle_width / 2,
                        -self.body_half_size[1] - self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_-",
                    pos=[
                        -self.handle_width / 2,
                        -self.body_half_size[1] - self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )

        main_body.append(handle_1)
        main_body.append(handle_2)
        main_body.append(
            new_site(
                name="pot_handle_1",
                rgba=self.rgba_handle_1,
                pos=handle_1_center - np.array([0, 0.005, 0]),
                size=[0.005],
            )
        )
        main_body.append(
            new_site(
                name="pot_handle_2",
                rgba=self.rgba_handle_2,
                pos=handle_2_center + np.array([0, 0.005, 0]),
                size=[0.005],
            )
        )
        main_body.append(new_site(name="pot_center", pos=[0, 0, 0], rgba=[1, 0, 0, 0]))

        return main_body

    def handle_geoms(self):
        return self.handle_1_geoms() + self.handle_2_geoms()

    def handle_1_geoms(self):
        if self.solid_handle:
            return ["handle_1"]
        return ["handle_1_c", "handle_1_+", "handle_1_-"]

    def handle_2_geoms(self):
        if self.solid_handle:
            return ["handle_2"]
        return ["handle_2_c", "handle_2_+", "handle_2_-"]

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


def five_sided_box(size, rgba, group, thickness):
    """
    Args:
        size ([float,flat,float]):
        rgba ([float,float,float,float]): color
        group (int): Mujoco group
        thickness (float): wall thickness

    Returns:
        []: array of geoms corresponding to the
            5 sides of the pot used in BaxterLift
    """
    geoms = []
    x, y, z = size
    r = thickness / 2
    geoms.append(
        new_geom(
            geom_type="box", size=[x, y, r], pos=[0, 0, -z + r], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[x, r, z], pos=[0, -y + r, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[x, r, z], pos=[0, y - r, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[r, y, z], pos=[x - r, 0, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[r, y, z], pos=[-x + r, 0, 0], rgba=rgba, group=group
        )
    )
    return geoms


DEFAULT_DENSITY_RANGE = [200, 500, 1000, 3000, 5000]
DEFAULT_FRICTION_RANGE = [0.25, 0.5, 1, 1.5, 2]


def _get_size(size,
              size_max,
              size_min,
              default_max,
              default_min):
    """
        Helper method for providing a size,
        or a range to randomize from
    """
    if len(default_max) != len(default_min):
        raise ValueError('default_max = {} and default_min = {}'
                         .format(str(default_max), str(default_min)) +
                         ' have different lengths')
    if size is not None:
        if (size_max is not None) or (size_min is not None):
            raise ValueError('size = {} overrides size_max = {}, size_min = {}'
                             .format(size, size_max, size_min))
    else:
        if size_max is None:
            size_max = default_max
        if size_min is None:
            size_min = default_min
        size = np.array([np.random.uniform(size_min[i], size_max[i])
                         for i in range(len(default_max))])
    return size


def _get_randomized_range(val,
                          provided_range,
                          default_range):
    """
        Helper to initialize by either value or a range
        Returns a range to randomize from
    """
    if val is None:
        if provided_range is None:
            return default_range
        else:
            return provided_range
    else:
        if provided_range is not None:
            raise ValueError('Value {} overrides range {}'
                             .format(str(val), str(provided_range)))
        return [val]


class BoundingObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object for sawyer fit
    """

    def __init__(
        self,
        size=[0.1,0.1],
        hole_size = [0.05,0.05,0.05],
        tolerance = 1.03,
        offset = [0.1,0.05,0.4]
    ):
        super().__init__()
        self.size = size
        self.hole_size = tolerance*np.asarray(hole_size)
        self.offset=np.asarray(offset)
    def get_bottom_offset(self):
        return np.array([0, 0, 0])

    def get_top_offset(self):
        return np.array([0, 0, self.size])

    def get_horizontal_radius(self):
        return np.sqrt(2) * (self.size)

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)
        x = self.size[0]-self.hole_size[0]
        y = self.size[1]-self.hole_size[1]
        x1 = np.random.uniform(x/5,4*x/5)
        x2 = x-x1
        y1 = np.random.uniform(y/5,4*y/5)
        y2 = y-y1
        main_body.append(
        new_geom(
            geom_type="box", size=[x1, self.size[1],self.hole_size[2]],pos=self.offset+[self.size[0]-x1, 0.0, self.hole_size[2]], group=1,
            material="lego", rgba=None)
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[x2, self.size[1],self.hole_size[2]],pos=self.offset+[-self.size[0]+x2, 0.0, self.hole_size[2]], group=1,
            material="lego", rgba=None)
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.size[0],y1,self.hole_size[2]],pos=self.offset+[0.0,-self.size[1]+y1, self.hole_size[2]], group=1,
            material="lego", rgba=None)
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.size[0],y2,self.hole_size[2]],pos=self.offset+[0.0,self.size[1]-y2, self.hole_size[2]], group=1,
            material="lego", rgba=None)
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def in_grid(self,point,size):
        # checks if an object centered at point of dimensions size is within the hole
        result = True
        x,y,z = point
        if not (x -size[0] > self.offset[0]-self.size[0] and x + size[0] < self.offset[0]+self.size[0]):
            result = False
        if not (y-size[1] > self.offset[1]-self.size[1] and y + size[1] < self.offset[1]+self.size[1]):
            result = False
        if not (z - size[2] > self.offset[2] and z + size[2] < 1.1*(self.offset[2]+2*self.hole_size[2])):
            #Hack for z tolerance
            #print("z",self.offset[2],self.offset[2]+2*self.hole_size[2])
            result = False
        return result

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


class HoleObject(MujocoGeneratedObject):
    """
    Generates 2d lego brick
    """

    def __init__(
        self,
        size=0.01,
        tolerance = 0.98,
        pattern = [[1,1,1],[1,0,1]],
        z_compress = 1.0,
        name = ''    ):
        super().__init__()
        self.size = tolerance*size
        self.pattern = pattern
        self.z_compress = z_compress
        self.name = name
    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size])

    def get_top_offset(self):
        return np.array([0, 0, self.size])

    def get_horizontal_radius(self):
        return np.sqrt(2) * (self.size)

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)

        pattern = self.pattern
        cnt = 0
        for i in range(len(pattern)):
            for j in range(len(pattern[0])):
                mat = 'lego1'
                if self.name =='1':
                    mat = 'lego'
                if(pattern[i][j]):
                    main_body.append(
                    new_geom(
                        geom_type="box", size=[self.size, self.size, self.z_compress*self.size], pos=[2*i*self.size-self.size*len(pattern), 2*j*self.size-self.size*len(pattern), 0.0], group=1,
                        material=mat, rgba=None)
                    )
                    main_body[-1].set('name','block'+self.name+'-'+str(cnt))
                    cnt +=1
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


class Hole3dObject(MujocoGeneratedObject):
    """
    Generates the 3d grid object for assembly task
    """

    def __init__(
        self,
        size=0.01,
        pattern=[[1,1,1,1,1,1,1],[1,1,0,1,1,1,1],[1,1,0,0,0,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1]],
        offset =0,
        z_compress = 1.0):
        super().__init__()
        self.size = size
        self.pattern = pattern
        self.offset = offset
        self.z_compress = z_compress
    def get_bottom_offset(self):
        return np.array([0, 0, 0])

    def get_top_offset(self):
        return np.array([0, 0, self.size*2*len(self.pattern)])

    def get_horizontal_radius(self):
        return np.sqrt(2) * (4*self.size)

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)
        pattern = self.pattern
        for k in range(len(pattern)):
            for i in range(len(pattern[0])):
                for j in range(len(pattern[0][0])):
                    if(pattern[k][i][j]>0):
                        mat = 'lego'
                        if k < len(pattern)-1 and i > 0 and j > 0 and i < len(pattern[0]) -1 and j < len(pattern[0][0])-1 :
                            mat = 'lego1'
                        main_body.append(
                        new_geom(
                            geom_type="box", size=[pattern[k][i][j]*self.size,pattern[k][i][j]*self.size, self.z_compress*self.size], pos=[self.offset+2*i*self.size-self.size*len(pattern[0]), self.offset+2*j*self.size-self.size*len(pattern[0][0]),self.size+2*k*self.z_compress*self.size], group=1,
                            material=mat, rgba=None, density='100')
                        )

        return main_body

        # Check if point is within the grid
    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


class GridObject(MujocoGeneratedObject):
    """
    Generates the hole object
    """

    def __init__(
        self,
        size=0.01,
        pattern=[[1,1,1,1,1,1,1],[1,1,0,1,1,1,1],[1,1,0,0,0,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1]],
        offset =0,
        z_compress = 1.0):
        super().__init__()
        self.size = size
        self.pattern = pattern
        self.offset = offset
        self.z_compress = z_compress
    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size])

    def get_top_offset(self):
        return np.array([0, 0, self.size])

    def get_horizontal_radius(self):
        return np.sqrt(2) * (4*self.size)

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)
        pattern = self.pattern
        for k in range(len(pattern)):
            for i in range(len(pattern[0])):
                for j in range(len(pattern[0][0])):
                    if(pattern[k][i][j]>0):
                        mat = 'lego'
                        if k < len(pattern)-1 and i > 0 and j > 0 and i < len(pattern[0]) -1 and j < len(pattern[0][0])-1 :
                            mat = 'lego1'
                        main_body.append(
                        new_geom(
                            geom_type="box", size=[pattern[k][i][j]*self.size,pattern[k][i][j]*self.size, self.z_compress*self.size], pos=[self.offset+2*i*self.size-self.size*len(pattern[0]), self.offset+2*j*self.size-self.size*len(pattern[0][0]), 0.4+self.size+2*k*self.z_compress*self.size], group=1,
                            material=mat, rgba=None, density='100')
                        )

        return main_body
    def in_grid(self,point):
        # checks if point is within the hole
        result = True
        pattern = self.pattern
        x,y,z = point
        if not (x > self.offset-self.size*len(pattern[0]) and x < self.offset+self.size*len(pattern[0])):
            result = False
        if not (y > self.offset-self.size*len(pattern[0][0]) and y < self.offset+self.size*len(pattern[0][0])):
            result = False
        if not (z > 0.4 and z < 0.4+2*len(pattern)*self.z_compress*self.size):
            result = False
        return result

        # Check if point is within the grid
    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


class BoxObject(MujocoGeneratedObject):
    """
    An object that is a box
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07, 0.07],
                         [0.03, 0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 3, "box size should have length 3"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.size[0:2], 2)

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="box")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="box")


class CylinderObject(MujocoGeneratedObject):
    """
    A randomized cylinder object.
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07],
                         [0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 2, "cylinder size should have length 2"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[1]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[1]])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="cylinder")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="cylinder")


class BallObject(MujocoGeneratedObject):
    """
    A randomized ball (sphere) object.
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07],
                         [0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 1, "ball size should have length 1"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[0]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[0]])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="sphere")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="sphere")


class CapsuleObject(MujocoGeneratedObject):
    """
    A randomized capsule object.
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07],
                         [0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 2, "capsule size should have length 2"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * (self.size[0] + self.size[1])])

    def get_top_offset(self):
        return np.array([0, 0, (self.size[0] + self.size[1])])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="capsule")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="capsule")


### More Miscellaneous Objects ###


class AnimalObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.02,0.033)
        self.body_y = random.uniform(0.015,0.03)
        self.body_z = random.uniform(0.01,0.035)
        self.legs_x = random.uniform(0.005,0.01)
        self.legs_z = random.uniform(0.01,0.035)
        self.neck_x = random.uniform(0.005,0.01)
        self.neck_z = random.uniform(0.005,0.01)
        self.head_y = random.uniform(0.010,0.015)
        self.head_z = random.uniform(0.005,0.01)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-2*self.legs_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.neck_z+2*self.head_z])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+self.body_y**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #legs
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[0.9*self.body_x-self.legs_x, 0.9*self.body_y-self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[-0.9*self.body_x+self.legs_x, 0.9*self.body_y-self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[0.9*self.body_x-self.legs_x, -0.9*self.body_y+self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[-0.9*self.body_x+self.legs_x, -0.9*self.body_y+self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1),)
        )
        #neck
        main_body.append(
        new_geom(
            geom_type="box", size=[self.neck_x,self.neck_x,self.neck_z],pos=[self.body_x-self.neck_x, 0, self.neck_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="box", size=[self.head_y,self.neck_x*1.5,self.head_z],pos=[self.body_x-2*self.neck_x+self.head_y, 0, 2*self.neck_z+self.body_z+self.head_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class CarObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.02,0.031)
        self.body_y = random.uniform(0.015,0.03)
        self.body_z = random.uniform(0.01,self.body_x/2)
        self.wheels_r = random.uniform(self.body_x/4.0,self.body_x/3.0)
        self.wheels_z = random.uniform(0.002,0.004)
        self.top_x = random.uniform(0.008,0.9*self.body_x)
        self.top_y = random.uniform(0.007,0.9*self.body_y)
        self.top_z = random.uniform(0.004,0.9*self.body_z)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-self.wheels_r])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.top_z])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+(self.body_y+2*self.wheels_z)**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #wheels
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1),)
        )
        #top
        main_body.append(
        new_geom(
            geom_type="box", size=[self.top_x,self.top_y,self.top_z],pos=[0, 0, self.top_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )

        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class TrainObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.017,0.031)
        self.body_y = random.uniform(0.025,0.045)
        self.body_z = random.uniform(0.01,0.025)
        self.wheels_r = random.uniform(self.body_x/4.0,self.body_x/3.0)
        self.wheels_z = random.uniform(0.002,0.006)
        self.top_x = random.uniform(0.01,0.9*self.body_x)
        self.top_r = 0.99*self.body_x
        self.top_z = 0.99*self.body_y
        self.cabin_x = 0.99*self.body_x
        self.cabin_y = random.uniform(0.20,0.3)*self.body_y
        self.cabin_z = random.uniform(0.5,0.8)*self.top_r
        self.chimney_r = random.uniform(0.004,0.01)
        self.chimney_z = random.uniform(0.01,0.03)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-self.wheels_r])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.chimney_z+self.top_r])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+(self.body_y+2*self.wheels_z)**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #wheels
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1),)
        )
        #top
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.top_r,self.top_z],pos=[0, 0, self.body_z], group=1, zaxis="0 1 0",
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #cabin
        main_body.append(
        new_geom(
            geom_type="box", size=[self.cabin_x,self.cabin_y,self.cabin_z],pos=[0, -self.body_y+self.cabin_y, self.body_z+self.cabin_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #chimney
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.chimney_r,self.chimney_z],pos=[0, self.body_y*.5, self.body_z+self.top_r], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class BipedObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.02,0.031)
        self.body_y = random.uniform(0.017,0.022)
        self.body_z = random.uniform(0.015,0.03)
        self.legs_x = random.uniform(0.005,0.01)
        self.legs_z = random.uniform(0.005,self.body_z)
        self.hands_x = random.uniform(0.005,0.01)
        self.hands_z = random.uniform(0.01,0.3*self.legs_z)
        self.head_y = self.body_y
        self.head_z = random.uniform(0.01,0.02)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-2*self.legs_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+self.body_y**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #legs
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.body_y,self.legs_z],pos=[self.body_x-self.legs_x, 0, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.body_y,self.legs_z],pos=[-self.body_x+self.legs_x, 0, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )

        #hands
        main_body.append(
        new_geom(
            geom_type="box", size=[self.hands_x,2*self.body_y,self.hands_z],pos=[self.body_x+self.hands_x, self.body_y, -self.hands_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.hands_x,2*self.body_y,self.hands_z],pos=[-self.body_x-self.hands_x, self.body_y, -self.hands_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="box", size=[self.head_y,self.head_y,self.head_z],pos=[0, 0, self.body_z+self.head_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


class DumbbellObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_r = random.uniform(0.009,0.013)
        self.body_z = random.uniform(0.015,0.025)
        self.head_r = random.uniform(1.6*self.body_r,2*self.body_r)
        self.head_z = random.uniform(0.005,0.01)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-2*self.head_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return self.body_z+self.head_z

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.body_r,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, -self.head_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, self.head_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )        

        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class HammerObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_r = random.uniform(0.009,0.013)
        self.body_z = random.uniform(0.027,0.037)
        self.head_r = random.uniform(1.6*self.body_r,3*self.body_r)
        self.head_z = random.uniform(1.5*self.body_r,2*self.body_r)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return self.body_r+self.head_r

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.body_r,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, 0.95*self.head_r+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1),zaxis='1 0 0')
        )
    

        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


class GuitarObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_r = random.uniform(0.021,0.027)/1.7
        self.body_z = random.uniform(0.017,0.025)/1.4
        self.head_r = random.uniform(1.5,2)*self.body_r
        self.head_z = self.body_z
        self.arm_x = random.uniform(0.008,0.010)/2
        self.arm_y = random.uniform(1.2,1.6)*(self.body_r+self.head_r)
        self.arm_z = 0.007/2
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return self.body_r+self.head_r

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        color = np.append(np.random.uniform(size=3),1)
        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.body_r,self.body_z],pos=[0, self.head_r+0.5*self.body_r, 0], group=1,
             rgba=color)
        )
        #head
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, 0], group=1,
             rgba=color)
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r*0.5,self.head_z],pos=[0, 0, 0.001], group=1,
             rgba=[0,0,0,1])
        )
        #arm
        main_body.append(
        new_geom(
            geom_type="box", size=[self.arm_x,self.arm_y,self.arm_z],pos=[0, self.arm_y, self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


