# Objects

## How to create a custom object
Objects, such as boxes and cans, are essential to building manipulation environments. We designed the [MujocoObject](../source/robosuite.models.objects.html#robosuite.models.objects.objects.MujocoObject) interfaces to standardize and simplify the procedure for importing 3D models into the scene or procedurally generate new objects. MuJoCo defines models via the [MJCF](http://www.mujoco.org/book/modeling.html) XML format. These MJCF files can either be stored as XML files on disk and loaded into simulator, or be created on-the-fly by code prior to simulation. Based on these two mechanisms of how MJCF models are created, we offer two main ways of creating your own object:

* Define an object in an MJCF XML file;
* Use procedural generation APIs to dynamically create an MJCF model.

## The MujocoObject class
```python
class MujocoObject(MujocoModel):
    def __init__(...):
        
        ...

        # Attributes that should be filled in within the subclass
        self._name = None
        self._obj = None

        # Attributes that are auto-filled by _get_object_properties call
        self._root_body = None
        self._bodies = None
        self._joints = None
        self._actuators = None
        self._sites = None
        self._contact_geoms = None
        self._visual_geoms = None
```
`MujocoObject` is the base class of all objects. One must note that it is not a subclass of `MujocoXML`, but does extend from the unifying `MujocoModel` class from which all simulation models (including robots, grippers, etc.) should extend from. All of the attributes shown above prepended with a `_` are intended to be private variables and not accessed by external objects. Instead, any of these properties can be accessed via its public version, without the `_` (e.g.: to access all the object's joints, call `obj.joints` instead of `obj._joints`). This is because all public attributes are automatically post-processed from their private counterparts and have naming prefixes appended to it.

The XML of an object is generated once during initialization via the `_get_object_subtree` call, after which any external object can extract a reference to this XML via the `get_obj` call.
```python
    def _get_object_subtree(self):
        pass

    def get_obj(self):
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
pos = table_top - bottom_offset                             # pos + bottom_offset = table_top
obj_xml = obj.get_obj().set("pos", array_to_string(pos))    # Set the top-level body of this object
```

## Creating a XMLObject
One can use MuJoCo MJCF XML to generate an object, either as a composition of primitive [geoms](http://mujoco.org/book/modeling.html#geom) or imported from STL files of triangulated [meshes](http://www.mujoco.org/book/modeling.html#mesh). An example is `robosuite.models.objects.xml_objects.BreadObject`. Its [python definition](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/objects/xml_objects.py#L49) is short. Note that all `MujocoXMLObject` classes require both a `fname` and `name` argument, the former which specifies the filepath to the raw XML file and the latter which specifies the in-sim name of the object instantiated. The optional `joints` argument can also specify a custom set of joints to apply to the given object (defaults to "default", which is a single free joint). This joint argument determines the DOF of the object as a whole and does not interfere with the joints already in the object. Additionally, the type of object created can be specified via the `obj_type` argument, and must be one of (`'collision'`, `'visual'`, or `'all'`). Lastly, setting `duplicate_collision_geoms` makes sure that all collision geoms automatically have an associated visual geom as well. Generally, the normal use case is to define a single class corresponding to a specific XML file, as shown below:
```python
class BreadObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(xml_path_completion("objects/bread.xml"),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)
```

In the corresponding XML file, a few key definitions must be present. The top-level, un-named body must contain as immediate children tags (a) the actual object bodie(s) (the top-level **must** be named `object`) and (b) three site tags named `bottom_site`, `top_site`, and `horizontal_radius_site` and whose `pos` values must be specified. The example for the `BreadObject`, [bread.xml](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/assets/objects/bread.xml), is shown below:
```xml
<mujoco model="bread">
  <asset>
    <mesh file="meshes/bread.stl" name="bread_mesh" scale="0.8 0.8 0.8"/>
    <texture file="../textures/bread.png" type="2d" name="tex-bread" />
    <material name="bread" reflectance="0.7" texrepeat="15 15" texture="tex-bread" texuniform="true"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom pos="0 0 0" mesh="bread_mesh" type="mesh" solimp="0.998 0.998 0.001" solref="0.001 1" density="50" friction="0.95 0.3 0.1"  material="bread" group="0" condim="4"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 -0.045" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0.03" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0.03 0.03 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
```
Concretely,
* `_get_object_subtree` looks for the object bodie(s) as defined by all nested bodie(s) beginning with the `object`-named body tag.
* `bottom_site` should be the bottom of the object, i.e. contact point with the surface it is placed on.
* `top_site` should be the top of the object, i.e. contact point if something is placed on it.
* `horizontal_radius_site` can be any point on a circle in the x-y plane that does not intersect the object. This allows us to place multiple objects without having them collide into one another.

## Creating a procedurally generated object
Procedurally generated objects have been used in [several](https://arxiv.org/abs/1802.09564) [recent](https://arxiv.org/abs/1806.09266) [works](https://arxiv.org/abs/1709.07857) to train control policies with improved robustness and generalization. Here you can programmatically generate an MJCF XML of an object from scratch using `xml.etree.ElementTree`, and compose an object of multiple geom primitives. The base class for this type of object is `MujocoGeneratedObject`.
**robosuite** natively supports all Mujoco primitive objects with procedurally-generated `PrimitiveObject` classes (`BoxObject`, `BallObject`, `CapsuleObject`, and `CylinderObject`).

Additionally, **robosuite** supports custom, complex objects that can be defined by collections of primitive geoms (the [CompositeObject](../source/robosuite.models.objects.html#robosuite.models.objects.generated_objects.CompositeObject) class) or even other objects (the [CompositeBodyObject](../source/robosuite.models.objects.html#robosuite.models.objects.generated_objects.CompositeBodyObject) class). The APIs for each of these classes have been standardized for ease of usage, and interested readers should consult the docstrings for each of these classes, as well as provided examples of each class ([HammerObject](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/objects/composite/hammer.py#L10), [HingedBoxObject](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/objects/composite_body/hinged_box.py#L8)).

It should also be noted that all of the above classes extending from the `MujocoGenereatedObject` class automatically supports custom texture definitions on a per-geom level, where specific texture images can be mapped to individual geoms. The above `HammerObject` showcases an example applying custom textures to different geoms of the resulting object.

## Placing Objects

Object locations are initialized on every environment reset using instances of the [ObjectPositionSampler](../source/robosuite.utils.html#robosuite.utils.placement_samplers.ObjectPositionSampler) class. Object samplers use the `bottom_site` and `top_site` sites of each object in order to place objects on top of other objects, and the `horizontal_radius_site` site in order to ensure that objects do not collide with one another. The most basic sampler is the [UniformRandomSampler](../source/robosuite.utils.html#robosuite.utils.placement_samplers.UniformRandomSampler) class - this just uses rejection sampling to place objects randomly. As an example, consider the following code snippet from the `__init__` method of the `Lift` environment class.

```python
self.placement_initializer = UniformRandomSampler(
    name="ObjectSampler",
    mujoco_objects=self.cube,
    x_range=[-0.03, 0.03],
    y_range=[-0.03, 0.03],
    rotation_axis='z',
    rotation=None,
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=True,
    reference_pos=self.table_offset,
    z_offset=0.01,
)
```

This will sample the `self.cube`'s object location uniformly at random in a box of size `0.03` (`x_range`, `y_range`) with random (`rotation`) z-rotation (`rotation_axis`), and with an offset of `0.01` (`z_offset`) above the table surface location (`reference_pos`). The sampler will also make sure that the entire object boundary falls within the sampling box size (`ensure_object_boundary_in_range`) and does not collide with any placed objects (`ensure_valid_placement`).

Another common sampler is the [SequentialCompositeSampler](../source/robosuite.utils.html#robosuite.utils.placement_samplers.SequentialCompositeSampler), which is useful for composing multiple arbitrary placement samplers together. As an example, consider the following code snippet from the `__init__` method of the `NutAssembly` environment class. 

```python
# Establish named references to each nut object
nut_names = ("SquareNut", "RoundNut")

# Initialize the top-level sampler
self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

# Create individual samplers per nut
for nut_name, default_y_range in zip(nut_names, ([0.11, 0.225], [-0.225, -0.11])):
    self.placement_initializer.append_sampler(
        sampler=UniformRandomSampler(
            name=f"{nut_name}Sampler",
            x_range=[-0.115, -0.11],
            y_range=default_y_range,
            rotation=None,
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.02,
        )
    )
   
# No objects have been assigned to any samplers yet, so we do that now
for i, (nut_cls, nut_name) in enumerate(zip(
        (SquareNutObject, RoundNutObject),
        nut_names,
)):
    nut = nut_cls(name=nut_name)
    self.placement_initializer.add_objects_to_sampler(sampler_name=f"{nut_name}Sampler", mujoco_objects=nut)
```

The code snippet above results in two `UniformRandomSampler` instances being used to place the nuts onto the table surface - one for each type of nut. Notice this also allows the nuts to be initialized in separate regions of the table, and with arbitrary sampling settings. The `SequentialCompositeSampler` makes it easy to compose multiple placement initializers together and assign objects to each sub-sampler in a modular way.