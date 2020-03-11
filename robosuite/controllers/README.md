# Controllers  
  Controllers are used to determine the type of high-level control over a given robot arm. While all arms are directly controlled via their joint torques, the inputted action space for a given environment can vary depending on the type of desired control. Below, a list of supported controllers and their respective action dimensions are listed.
  |   Controller   					| 			Action Space (not including gripper)    |  
| :-----------------------------------------------: | :--------------------------------:|  
|    End Effector Position Orientation (EE_POS_ORI) |    6 (x, y, z, r, p, y)          	|  
|    End Effector Position (EE_POS) 				|    3 (x, y, z)      				|  
|    End Effector Inverse Kinematics (EE_IK)  		|    7 (x, y, z, i, j, k, w) 		|  
|    Joint Position (JOINT_IMP)    					|    n (robot joint dimension)      |  
|    Joint Velocity (JOINT_VEL)    					|    n (robot joint dimension)      |  
|    Joint Torque (JOINT_TORQUE)   					|    n (robot joint dimension)      |  

## Configurations
The [config directory](robosuite/controllers/config) provides a set of default configuration files that hold default examples of parameters relevant to individual controllers. Note that when creating your controller config templates of a certain type of controller, the listed parameters in the default example should be specified accordingly.

Below, a brief overview and description of each subset of controller parameters are shown:

#### General Controller Settings  
* `'type'`: Type of controller to control. Can be `'EE_POS_ORI'`, `'EE_POS'`, `'EE_IK'`, `'JOINT_IMP'`, `'JOINT_VEL'`, or `'JOINT_TORQUE'`
* `'interpolation'`: If not `null`, specified type of interpolation to use between desired actions. Currently only `'linear'` is supported. 
* `'{...}_limits'`: Limits for that specific controller. E.g.: for a `'JOINT_IMP'`, the relevant limits are its joint positions, `'qpos_limits'` . Can be either a 2-element list (same min/max limits across entire relevant space), or a list of lists (specific limits for each component)
* `'{input,output}_{min,max}'`: Scaling ranges for mapping action space inputs into controller inputs. Settings these limits will automatically clip the action space input to be within the `'input_{min,max}'` before mapping the requested value into the specified `'output_{min,max}'` range. Can be either a scalar (same limits across entire action space), or a list (specific limits for each action component)
* `'kp', 'kv'`: Where relevant, specifies the positional / velocity gain for the controller. Can be either be a scalar (same value for all robot joints), or a list (specific values for each joint)
* `'damping'`: Where relevant, specifies the damping constant for the controller.
* `'control_delta'`: Only relevant for `'EE_POS_ORI'` or `'EE_POS'` controllers. `true` interprets input actions as delta values from the current robot end effector position. Otherwise, assumed to be absolute (global) values
* `'uncouple_pos_ori'`: Only relevant for `'EE_POS_ORI'`. `true` decouples the desired position and orientation when executing the controller

## Using a Custom Controller Configuration
A custom controller other than the environment defaults (which are normally are the default `JOINT_VEL` configurations) can be used by simply creating a new config (`.json`) file with the relevant parameters as specified above. All robosuite environments have an optional `controller_config` argument that can be used to pass in specific controller settings. Note that this is expected to be a `dict`, so the new configuration must be read in and parsed as a `dict` before passing it during the environment `robosuite.make(...)` call. A brief example script showing how to import a custom controller configuration is shown below.

```python
import robosuite as suite
import json

# Path to config file
controller_fpath = `{your_custom_config_filepath_here}`

# Import the file as a dict
with open(controller_fpath) as f:
	config = json.load(f)

# Create environment
env = suite.make("PandaLift", controller_config=config, ... )
```