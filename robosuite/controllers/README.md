# Controllers  
  Controllers are used to determine the type of high-level control over a given robot arm. While all arms are directly controlled via their joint torques, the inputted action space for a given environment can vary depending on the type of desired control. Below, a list of supported controllers and their respective action dimensions are listed.
| Controller Name |   Controller Type			| 			Action Dimension<br>(Gripper Not Included)    |  Format |  
| :-------------: | :-------------------: | :-------------------------------------------------: | :-----: |
| EE_OSC_POSE     |   End Effector Position + Orientation via Operational Space Control | 6 | (x, y, z, r, p, y)    |  
| EE_OSC_POSITION |   End Effector Position via Operational Space Control               | 3 | (x, y, z)             |  
| EE_IK_POSE      |   End Effector Position + Orientation via Inverse Kinematics        | 7 | (x, y, z, i, j, k, w) |
| JOINT_POSITION  |   Joint Position                                                    | n | n robot joints        |  
| JOINT_VELOCITY  |   Joint Velocity                                                    | n | n robot joints        |  
| JOINT_TORQUE    |   Joint Torque             					                        | n | n robot joints        |  

## Configurations
The [config directory](config) provides a set of default configuration files that hold default examples of parameters relevant to individual controllers. Note that when creating your controller config templates of a certain type of controller, the listed parameters in the default example are required and should be specified accordingly.

Note: Each robot ([Sawyer](config/default_sawyer.json), [Panda](config/default_panda.json), [Baxter](config/default_baxter.json)) has its own default controller configuration which is called by default unless a [different controller config](#using-a-custom-controller-configuration) is called.

Below, a brief overview and description of each subset of controller parameters are shown:

#### General Controller Settings  
* `type`: Type of controller to control. Can be `EE_OSC_POSE`, `EE_OSC_POSITION`, `EE_IK_POSE`, `JOINT_POSITION`, `JOINT_VELOCITY`, or `JOINT_TORQUE`
* `interpolation`: If not `null`, specified type of interpolation to use between desired actions. Currently only `linear` is supported. 
* `{...}_limits`: Limits for that specific controller. E.g.: for a `JOINT_POSITION`, the relevant limits are its joint positions, `qpos_limits` . Can be either a 2-element list (same min/max limits across entire relevant space), or a list of lists (specific limits for each component)
* `ik_{pos, ori}_limit`: Only applicable for IK controller. Limits the magnitude of the desired relative change in position / orientation.
* `{input,output}_{min,max}`: Scaling ranges for mapping action space inputs into controller inputs. Settings these limits will automatically clip the action space input to be within the `input_{min,max}` before mapping the requested value into the specified `output_{min,max}` range. Can be either a scalar (same limits across entire action space), or a list (specific limits for each action component)
* `kp, kv`: Where relevant, specifies the positional / velocity gain for the controller. Can be either be a scalar (same value for all robot joints), or a list (specific values for each joint)
* `damping`: Where relevant, specifies the damping constant for the controller.
* `control_delta`: Only relevant for `EE_OSC_POSE` or `EE_OSC_POSITION` controllers. `true` interprets input actions as delta values from the current robot end effector position. Otherwise, assumed to be absolute (global) values
* `uncouple_pos_ori`: Only relevant for `EE_OSC_POSE`. `true` decouples the desired position and orientation torques when executing the controller

## Using a Custom Controller Configuration
A custom controller other than the environment defaults (which are normally are `JOINT_VELOCITY` configurations specified for each robot) can be used by simply creating a new config (`.json`) file with the relevant parameters as specified above. All robosuite environments have an optional `controller_config` argument that can be used to pass in specific controller settings. Note that this is expected to be a `dict`, so the new configuration must be read in and parsed as a `dict` before passing it during the environment `robosuite.make(...)` call. A brief example script showing how to import a custom controller configuration is shown below.

```python
import robosuite as suite
from robosuite.controllers import load_controller_config

# Path to config file
controller_fpath = `/your/custom/config/filepath/here/filename.json`

# Import the file as a dict
config = load_controller_config(custom_fpath=controller_fpath)

# Create environment
env = suite.make("PandaLift", controller_config=config, ... )
```

Alternatively, you can load a default controller with the following substitution, where `controller_name` is one of acceptable controller `type` strings:
```python
config = load_controller_config(default_controller=controller_name)
```
