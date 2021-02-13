# Dynamics Randomization

In order to achieve reasonable runtime speeds, many physics simulation platforms often must simplify the underlying physics model. Mujoco is no different, and as a result, many parameters such as friction, damping, and contact constraints do not fully capture real-world dynamics.

To better compensate for this, **robosuite** provides the `DynamicsModder` class, which can control individual dynamics parameters for each model within an environment. Theses parameters are sorted by element group, and briefly described below (for more information, please see [Mujoco XML Reference](http://www.mujoco.org/book/XMLreference.html)):
 
#### Opt (Global) Parameters
- `density`: Density of the medium (i.e.: air)
- `viscosity`: Viscosity of the medium (i.e.: air)

#### Body Parameters
- `position`: (x, y, z) Position of the body relative to its parent body
- `quaternion`: (qw, qx, qy, qz) Quaternion of the body relative to its parent body
- `inertia`: (ixx, iyy, izz) diagonal components of the inertia matrix associated with this body
- `mass`: mass of the body

#### Geom Parameters
- `friction`: (sliding, torsional, rolling) friction values for this geom
- `solref`: (timeconst, dampratio) contact solver values for this geom
- `solimp`: (dmin, dmax, width, midpoint, power) contact solver impedance values for this geom

#### Joint parameters
- `stiffness`: Stiffness for this joint
- `frictionloss`: Friction loss associated with this joint
- `damping`: Damping value for this joint

This `DynamicsModder` follows the same basic API as the other `Modder` classes, and allows per-parameter and per-group randomization enabling. Apart from randomization, this modder can also be instantiated to selectively modify values at runtime. A brief example is given below:

```python
import robosuite as suite
from robosuite.utils.mjmod import DynamicsModder
import numpy as np

# Create environment and modder
env = suite.make("Lift", robots="Panda")
modder = DynamicsModder(sim=env.sim, random_state=np.random.RandomState(5))

# Define function for easy printing
cube_body_id = env.sim.model.body_name2id(env.cube.root_body)
cube_geom_ids = [env.sim.model.geom_name2id(geom) for geom in env.cube.contact_geoms]

def print_params():
    print(f"cube mass: {env.sim.model.body_mass[cube_body_id]}")
    print(f"cube frictions: {env.sim.model.geom_friction[cube_geom_ids]}")
    print()
    
# Print out initial parameter values
print("INITIAL VALUES")
print_params()

# Modify the cube's properties
modder.mod(env.cube.root_body, "mass", 5.0)                                # make the cube really heavy
for geom_name in env.cube.contact_geoms:
    modder.mod(geom_name, "friction", [2.0, 0.2, 0.04])           # greatly increase the friction
modder.update()                                                   # make sure the changes propagate in sim

# Print out modified parameter values
print("MODIFIED VALUES")
print_params()

# We can also restore defaults (original values) at any time
modder.restore_defaults()

# Print out restored initial parameter values
print("RESTORED VALUES")
print_params()
```

Running [demo_domain_randomization.py](../demos.html#domain-randomization) is another method for demo'ing (albeit an extreme example of) this functionality.

Note that the modder already has some sanity checks in place to prevent presumably undesired / non-sensical behavior, such as adding damping / frictionloss to a free joint or setting a non-zero stiffness value to a joint that is normally non-stiff to begin with.
