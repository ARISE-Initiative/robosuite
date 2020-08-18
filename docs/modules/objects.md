# Object

## How to create a custom object
Objects, such as boxes and cans, are essential to building manipulation environments. We designed the [MujocoObject](../robosuite/models/objects/objects.py) interfaces to standardize and simplify the procedure for importing 3D models into the scene or procedurally generate new objects. MuJoCo defines models via the [MJCF](http://www.mujoco.org/book/modeling.html) xml format. These MJCF files can either be stored as XML files on disk and loaded into simulator, or be created on-the-fly by code prior to simulation. Based on these two mechanisms of how MJCF models are created, we offer two main ways of creating your own object:

* Define an object in an MJCF XML file;
* Use procedural generation APIs to dynamically create an MJCF model.

## The MujocoObject class
```python
class MujocoObject:
    def __init__(...):
        pass
```
`MujocoObject` is the base class of all objects. One must note that it is not a subclass of `MujocoXML`. The XML of an object is generated through the call to `get_collision` and `get_visual`. Both calls take an argument `name` which assigns a name to the generated `<body>` tag to help locate the object in simulation. They also take argument `site`, which adds a site to the center of the object. A site is helpful for locating the object in simulation.
```python
    def get_collision(self, name=None, site=False):
        pass

    def get_visual(self, name=None, site=False):
        pass
```

Additionally, objects are usually placed relatively. For example, we want to put an object on a table or place a cube on top of another. Instance methods `get_bottom_offset`, `get_top_offset`, `get_horizontal_radius` provide the necessary information to place objects properly. 
```python
    def get_bottom_offset(self):
        pass

    def get_top_offset(self):
        pass

    def get_horizontal_radius(self):
        pass
```
This allows us to do things like the following.
```python
table_top = np.array([0, 1, 0])
bottom_offset = obj.get_bottom_offset()
pos = table_top - bottom_offset # pos + bottom_offset = table_top
obj_xml = obj.get_visual().set("pos", array_to_string(pos))
```

## Creating a XMLObject
One can use MuJoCo MJCF xml to generate an object, either as a composition of primitive [geoms](http://mujoco.org/book/modeling.html#geom) or imported from STL files of triangulated [meshes](http://www.mujoco.org/book/modeling.html#mesh). An example is `robosuite.models.objects.xml_objects.BreadObject`. Its [python definition](../robosuite/models/objects/xml_objects.py#L41) is short.
```python
class BreadObject(MujocoXMLObject):
    def __init__(self):
        super().__init__(xml_path_completion("objects/bread.xml"))
```

So all the important definitions are in the [bread.xml](../robosuite/models/assets/objects/bread.xml) file.
```xml
<mujoco model="bread">
  <asset>
    <mesh file="meshes/bread.stl" name="bread_mesh" scale="0.8 0.8 0.8"/>
    <texture file="../textures/bread.png" type="2d" name="tex-bread" />
    <material name="bread" reflectance="0.7" texrepeat="15 15" texture="tex-bread" texuniform="true"/>
  </asset>
  <worldbody>
    <body>
      <body name="collision">
        <geom pos="0 0 0" mesh="bread_mesh" type="mesh" solimp="0.998 0.998 0.001" solref="0.001 1" density="50" friction="0.95 0.3 0.1"  material="bread" group="1" condim="4"/>
      </body>
      <body name="visual">
        <geom pos="0 0 0" mesh="bread_mesh" type="mesh" material="bread"  conaffinity="0" contype="0"  group="0" mass="0.0001"/>
        <geom pos="0 0 0" mesh="bread_mesh" type="mesh" material="bread"  conaffinity="0" contype="0"  group="1" mass="0.0001"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 -0.045" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0.03" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0.03 0.03 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
```
* `get_collision` of `MujocoXmlObject` would look for the `<body>` tag with name `collision`. 
* `get_visual` of `MujocoXmlObject` would look for the `<body>` tag with name `visual`.
* `bottom_site` should be the bottom of the object, i.e. contact point with the surface it is placed on.
* `top_site` should be the top of the object, i.e. contact point if something is placed on it.
* `horizontal_radius_site` can be any point on a circle in the x-y plane that does not intersect the object. This allows us to place multiple objects without having them collide into one another.
* These attributes will be parsed by the `MujocoXMLObject` class and conform to the `MujocoObject` interface.

## Creating a procedurally generated object
Proceudally generated objects have been used in [several](https://arxiv.org/abs/1802.09564) [recent](https://arxiv.org/abs/1806.09266) [works](https://arxiv.org/abs/1709.07857) to train control policies with improved robustness and generalization. Here you can programmatically generate a MJCF xml of an object from scratch using `xml.etree.ElementTree`. The implementation is straightforward and interested readers should refer to `_get_collision` and `get_visual` method of `MujocoGeneratedObject`, defined [here](../robosuite/models/objects/generated_objects.py). For instance, the `BaxterLift` environment used a pot with two side handles. This object is created with the procedural generation APIs, see [here](../robosuite/models/objects/generated_objects.py#L8).
