Models
======

Models provide a toolkit of composing modularized elements into a scene. The central goal of models is to provide a set of modularized APIs that procedurally generate combinations of robots, arenas, and parameterized 3D objects, such that it enables us to learn control policies with better robustness and generalization.

In MuJoCo, these scene elements are specified in [MJCF](http://mujoco.org/book/modeling.html#Summary) format and compiled into mjModel to instantiate the physical simulation. Each MJCF model can consist of multiple XML files as well as meshes and textures referenced from the XML file. These MJCF files are stored in the [assets](assets) folder. 

Below we describe the types of scene elements that we support:

Robots
------
The [robots](robots) folder contains robot classes which load robot specifications from MJCF/URDF files and instantiate a robot mjModel. All robot classes should inherit a base [Robot](robots/robot_model.py) class which defines a set of common robot APIs.

Grippers
--------
The [grippers](grippers) folder consists of a variety of end-effector models that can be mounted to the arms of a robot model by the [`add_gripper`](robots/robot_model.py#L20) method in the robot class.

Objects
-------
[Objects](objects) are small interactable scene elements that robots interact with using their actuators. Objects can be either defined as 3D [meshes](http://mujoco.org/book/modeling.html#mesh) (e.g., in STL format) or procedurally generated from primitive shapes of MuJoCo [geoms](http://mujoco.org/book/modeling.html#geom).

[MujocoObject](objects/mujoco_object.py) is the base object class. [MujocoXMLObject](objects/mujoco_object.py) is the base class for all objects that are loaded from MJCF XML files. [MujocoGeneratedObject](objects/mujoco_object.py) is the base class for procedurally generated objects with support for size and other physical property randomization.

Arenas
------
[Arenas](arenas) define the workspace, such as a tabletop or a set of bins, where the robot performs the tasks. All arena classes should inherit a base [Arena](arenas/arena.py) class. By default each arena contains 3 cameras (see [example](assets/arenas/empty_arena.xml)). The `frontview` camera provides an overview of the scene, which is often used to generate visualizations and video recordings. The `agentview` camera is the canonical camera typically used for visual observations when training visuomotor policies. The `birdview` camera is a top-down camera which is useful for debugging the placements of objects in the scene.

Tasks
-----
[Tasks](tasks) put together all the necessary scene elements, which typically consist of a robot, an arena, and a set of objects, into a model of the whole scene. It handles merging the MJCF models of individual elements and setting the initial placements of these elements in the scene. The resulting scene model is compiled and loaded into the MuJoCo backend to perform simulation. All task classes should inherit a base [Task](tasks/task.py) class which specifies a set of common APIs for model merging and placement initialization.
