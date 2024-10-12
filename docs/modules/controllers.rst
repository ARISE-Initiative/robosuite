Controllers
==============

Composite Controllers
---------------------

Basic
******


WholeBody IK
*************


ThirdParty Controllers
***********************


Workflow of Loading Configs
****************************
Loading configs for composite controllers is critical for selecting the correct control mode with well-tuned parameters. We provide a list of default controller configs for the composite controllers, and also support easy specification of your custom controller config file. A config file is defined in a json file. 

An example of the controller config file is shown below (many parameters are omitted in `...` for brevity):

.. toggle::

    Example for defining BASIC controller.

    .. code-block:: json

        {
        "type": "BASIC",
        "body_parts_controller_configs": {
            "arms": {
                "right": {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": 150,
                    ...
                },
                "left": {
                    "type": "OSC_POSE",
                    ...
                }
            },
            "torso": {
                "type" : "JOINT_POSITION",
                ...
            },
            "head": {
                "type" : "JOINT_POSITION",
                ...
            },
            "base": {
                "type": "JOINT_VELOCITY",
                ...
            },
            "legs": {
                "type": "JOINT_POSITION",
                ...
            }
            }
        }



Part Controllers
------------------ 

Part controllers are used to determine the type of high-level control over a given robot arm. While all arms are directly controlled via their joint torques, the inputted action space for a given environment can vary depending on the type of desired control. Our controller options include ``OSC_POSE``, ``OSC_POSITION``, ``JOINT_POSITION``, ``JOINT_VELOCITY``, and ``JOINT_TORQUE``.

For ``OSC_POSE``, ``OSC_POSITION``, and ``JOINT_POSITION``, we include three variants that determine the action space. The most common variant is to use a predefined and constant set of impedance parameters; in that case, the action space only includes the desired pose, position, or joint configuration. We also include the option to specify either the stiffness values (and the damping will be automatically set to values that lead to a critically damped system), or all impedance parameters, both stiffness and damping, as part of the action at each step. These two variants lead to extended action spaces that can control the stiffness and damping behavior of the controller in a variable manner, providing full control to the policy/solution over the contact and dynamic behavior of the robot.

When using any position-based control (``OSC``, ``IK``, or ``JOINT_POSITION`` controllers), input actions are, by default,
interpreted as delta values from the current state.

When using any end-effector pose controller (``IK``, ``OSC_POSE``), delta rotations from the current end-effector orientation
in the form of axis-angle coordinates ``(ax, ay, az)``, where the direction represents the axis and the magnitude
represents the angle (in radians). Note that for ``OSC``, the rotation axes are taken relative to the global world
coordinate frame, whereas for ``IK``, the rotation axes are taken relative to the end-effector origin, NOT the global world coordinate frame!

During runtime, the execution of the controllers is as follows. Controllers receive a desired configuration (reference value) and output joint torques that try to minimize the error between the current configuration and the desired one. Policies and solutions provide these desired configurations, elements of some action space, at what we call simulated policy frequency (:math:`f_{p}`), e.g., 20Hz or 30Hz. **robosuite** will execute several iterations composed of a controller execution and a simulation step at simulation frequency, :math:`f_s` (:math:`f_s = N\cdot f_p`), using the same reference signal until a new action is provided by the policy/solution. In these iterations, while the desired configuration is kept constant, the current state of the robot is changing, and thus, the error.

In the following we summarize the options, variables, and the control laws (equations) that convert desired values from the policy/solution and current robot states into executable joint torques to minimize the difference.

Joint Space Control - Torque
*********************************
Controller Type: ``JOINT_TORQUE``

Action Dimensions (not including gripper): ``n`` (number of joints)

Since our controllers transform the desired values from the policies/solutions into joint torques, if these values are already joint torques, there is a one-to-one mapping between the reference value from the policy/solution and the output value from the joint torque controller at each step: :math:`\tau = \tau_d`

.. math::
    \begin{equation}
    \tau = \tau_d
    \end{equation}

Joint Space Control - Velocity
*********************************
Controller Type: ``JOINT_VELOCITY``

Action Dimensions (not including gripper): ``n`` (number of joints)

To control joint velocities, we create a proportional (P) control law between the desired value provided by the policy/solution (interpreted as desired velocity of each joint) and the current joint velocity of the robot. This control law, parameterized by a proportional constant, :math:`k_p`, generates joint torques to execute at each simulation step:

.. math::
    \tau = k_p (\dot{q}_d - \dot{q})


Joint Space Control - Position with Fixed Impedance
********************************************************
Controller Type: ``JOINT_POSITION``

Impedance: fixed

Action Dimensions (not including gripper): ``n`` (number of joints)

In joint position control, we create a proportional-derivative (PD) control law between the desired value provided by the policy/solution (interpreted as desired configuration for each joint) and the current joint positions of the robot. The control law that generates the joint torques to execute is parameterized by proportional and derivative gains, :math:`k_p`` and :math:`k_v`, and defined as

.. math::
    \begin{equation}
    \tau = \Lambda \left[k_p \Delta_q - k_d\dot{q}\right]
    \end{equation} 

where :math:`\Delta_q  = q_d - q`` is the difference between current and desired joint configurations, and :math:`\Lambda`` is the inertia matrix, that we use to scale the error to remove the dynamic effects of the mechanism. The stiffness and damping parameters, :math:`k_p` and :math:`k_d`, are determined in construction and kept fixed.

Joint Space Control - Position with Variable Stiffness
***********************************************************
Controller Type: ``JOINT_POSITION``

Impedance: variable_kp

Action Dimensions (not including gripper): ``2n`` (number of joints)

The control law is the same as for fixed impedance but, in this controller, :math:`k_p`` can be determined by the policy/solution at each policy step.

Joint Space Control - Position with Variable Impedance
***********************************************************
Controller Type: ``JOINT_POSITION``

Impedance: variable

Action Dimensions (not including gripper): ``3n`` (number of joints)

Again, the control law is the same in the two previous control types, but now both the stiffness and damping parameters, :math:`k_p`` and :math:`k_d`, are controllable by the policy/solution and can be changed at each step.

Operational Space Control - Pose with Fixed Impedance
**********************************************************
Controller Type: ``OSC_POSE``

Impedance: fixed

Action Dimensions (not including gripper): ``6``

In the ``OSC_POSE`` controller, the desired value is the 6D pose (position and orientation) of a controlled frame. We follow the formalism from `[Khatib87] <https://ieeexplore.ieee.org/document/1087068>`_. Our control frame is always the ``eef_site`` defined in the [Gripper Model](../modeling/robot_model.html#gripper-model), placed at the end of the last link for robots without gripper or between the fingers for robots with gripper. The operational space control framework (OSC) computes the necessary joint torques to minimize the error between the desired and the current pose of the ``eef_site`` with the minimal kinematic energy. 

Given a desired pose :math:`\mathbf{x}_{\mathit{des}}` and the current end-effector pose, , we first compute the end-effector acceleration that would help minimize the error between both, assumed. PD (proportional-derivative) control schema to improve convergence and stability. For that, we first decompose into a desired position, :math:`p_d \in \mathbb{R}^3`, and a desired orientation, :math:`R_d \in \mathbb{SO}(3)`. The end-effector acceleration to minimize the error should increase with the difference between desired end-effector pose and current pose, :math:`p` and :math:`R` (proportional term), and decrease with the current end-effector velocity, :math:`v` and :math:`\omega` (derivative term).

We then compute the robot actuation (joint torques) to achieve the desired end-effector space accelerations leveraging the kinematic and dynamic models of the robot with the dynamically-consistent operational space formulation in [\[Khatib1995a\]](https://journals.sagepub.com/doi/10.1177/027836499501400103). First, we compute the wrenches at the end-effector that corresponds to the desired accelerations, :math:`{f}\in\mathbb{R}^{6}`.
Then, we map the wrenches in end-effector space :math:`{f}` to joint torque commands with the end-effector Jacobian at the current joint configuration :math:`J=J(q)`: :math:`\tau = J^T{f}`. 

Thus, the function that maps end-effector space position and orientation to low-level robot commands is (:math:`\textrm{ee} = \textrm{\it end-effector space}`):

.. math::

    \begin{equation}
    \begin{aligned}
    \tau &= J_p[\Lambda_p[k_p^p (p_d - p) - k_v^p v]] + J_R[\Lambda_R\left[k_p^R(R_d \ominus R) - k_d^R \omega \right]]
    \end{aligned}
    \end{equation}

where :math:`\Lambda_p` and :math:`\Lambda_R` are the parts corresponding to position and orientation in :math:`\Lambda \in \mathbb{R}^{6\times6}`, the inertial matrix in the end-effector frame that decouples the end-effector motions, :math:`J_p`` and :math:`J_R`` are the position and orientation parts of the end-effector Jacobian, and :math:`\ominus` corresponds to the subtraction in :math:`\mathbb{SO}(3)`. The difference between current and desired position (:math:`\Delta_p= p_d - p`) and between current and desired orientation (:math:`\Delta_R = R_d \ominus R`) can be used as alternative policy action space, :math:`\mathcal{A}`. :math:`k_p^p`, :math:`k_p^d`, :math:`k_p^R`, and :math:`k_d^R` are vectors of proportional and derivative gains for position and orientation (parameters :math:`\kappa`), respectively, set once at initialization and kept fixed.

Operational Space Control - Pose with Variable Stiffness
*************************************************************
Controller Type: ``OSC_POSE``

Impedance: variable_kp

Action Dimensions (not including gripper): ``12``

The control law is the same as ``OSC_POSE`` but, in this case, the stiffness of the controller, :math:`k_p`, is part of the action space and can be controlled and changed at each time step by the policy/solution. The damping parameters, :math:`k_d`, are set to maintain the critically damped behavior of the controller.

Operational Space Control - Pose with Variable Impedance
*********************************************************
Controller Type: ``OSC_POSE``

Impedance: variable

Action Dimensions (not including gripper): ``18``

The control law is the same as in the to previous controllers, but now both the stiffness and the damping, :math:`k_p` and :math:`k_d`, are part of the action space and can be controlled and changed at each time step by the policy/solution. 


Configurations
---------------
The config directory (found within the [Controllers](../source/robosuite.controllers.html) module) provides a set of default configuration files that hold default examples of parameters relevant to individual controllers. Note that when creating your controller config templates of a certain type of controller, the listed parameters in the default example are required and should be specified accordingly.

Note: Each robot has its own default controller configuration which is called by default unless a different controller config is called.

Below, a brief overview and description of each subset of controller parameters are shown:

Controller Settings  
********************
* ``type``: Type of controller to control. Can be ``OSC_POSE``, ``OSC_POSITION``, ``IK_POSE``, ``JOINT_POSITION``, ``JOINT_VELOCITY``, or ``JOINT_TORQUE``
* ``interpolation``: If not ``null``, specified type of interpolation to use between desired actions. Currently only ``linear`` is supported.
* ``ramp_ratio``: If using ``linear`` interpolation, specifies the proportion of allotted timesteps (value from [0, 1]) over which to execute the interpolated commands.
* ``{...}_limits``: Limits for that specific controller. E.g.: for a ``JOINT_POSITION``, the relevant limits are its joint positions, ``qpos_limits`` . Can be either a 2-element list (same min/max limits across entire relevant space), or a list of lists (specific limits for each component)
* ``ik_{pos, ori}_limit``: Only applicable for IK controller. Limits the magnitude of the desired relative change in position / orientation.
* ``{input,output}_{min,max}``: Scaling ranges for mapping action space inputs into controller inputs. Settings these limits will automatically clip the action space input to be within the ``input_{min,max}`` before mapping the requested value into the specified ``output_{min,max}`` range. Can be either a scalar (same limits across entire action space), or a list (specific limits for each action component)
* ``kp``: Where relevant, specifies the proportional gain for the controller. Can be either be a scalar (same value for all controller dimensions), or a list (specific values for each dimension)
* ``damping_ratio``: Where relevant, specifies the damping ratio constant for the controller.
* ``impedance_mode``: For impedance-based controllers (``OSC_*``, ``JOINT_POSITION``), determines the impedance mode for the controller, i.e. the nature of the impedance parameters. It can be ``fixed``, ``variable``, or ``variable_kp`` (kd is adjusted to provide critically damped behavior).
* ``kp_limits, damping_ratio_limits``: Only relevant if ``impedance_mode`` is set to ``variable`` or ``variable_kp``. Sets the limits for the resulting action space for variable impedance gains.
* ``control_delta``: Only relevant for ``OSC_POSE`` or ``OSC_POSITION`` controllers. ``true`` interprets input actions as delta values from the current robot end-effector position. Otherwise, assumed to be absolute (global) values
* ``uncouple_pos_ori``: Only relevant for ``OSC_POSE``. ``true`` decouples the desired position and orientation torques when executing the controller

Loading a Controller
---------------------
By default, if no controller configuration is specified during environment creation, then ``JOINT_VELOCITY`` controllers with robot-specific configurations will be used. 

Using a Default Controller Configuration
*****************************************
Any controller can be used with its default configuration, and can be easily loaded into a given environment by calling its name as shown below (where ``controller_name`` is one of acceptable controller ``type`` strings):

.. code-block:: python

    import robosuite as suite
    from robosuite import load_controller_config

    # Load the desired controller's default config as a dict
    config = load_controller_config(default_controller=controller_name)

    # Create environment
    env = suite.make("Lift", robots="Panda", controller_configs=config, ... )


Using a Custom Controller Configuration
****************************************
A custom controller configuration can also be used by simply creating a new config (``.json``) file with the relevant parameters as specified above. All robosuite environments have an optional ``controller_configs`` argument that can be used to pass in specific controller settings. Note that this is expected to be a ``dict``, so the new configuration must be read in and parsed as a ``dict`` before passing it during the environment ``robosuite.make(...)`` call. A brief example script showing how to import a custom controller configuration is shown below.


.. code-block:: python

    import robosuite as suite
    from robosuite import load_controller_config

    # Path to config file
    controller_fpath = "/your/custom/config/filepath/here/filename.json"

    # Import the file as a dict
    config = load_controller_config(custom_fpath=controller_fpath)

    # Create environment
    env = suite.make("Lift", robots="Panda", controller_configs=config, ... )