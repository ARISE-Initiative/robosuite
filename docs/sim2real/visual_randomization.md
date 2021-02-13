# Visual Randomization

It is well shown that randomizing the visuals in simulation can play an important role in sim2real applications. **robosuite** provides various `Modder` classes to control different aspects of the visual environment. This includes:

- `CameraModder`: Modder for controlling camera parameters, including FOV and pose
- `TextureModder`: Modder for controlling visual objects' appearances, including texture and material properties
- `LightingModder`: Modder for controlling lighting parameters, including light source properties and pose

Each of these Modders can be used by the user to directly override default simulation settings, or to randomize their respective properties mid-sim. We provide [demo_domain_randomization.py](../demos.html#domain-randomization) to showcase all of these modders being applied to randomize an environment during every sim step.
