"""
Script to showcase domain randomization functionality.
"""

import robosuite.utils.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import DomainRandomizationWrapper, GymImageDomainRandomizationWrapper

from robosuite.wrappers.domain_randomization_wrapper import DEFAULT_DYNAMICS_ARGS, DEFAULT_LIGHTING_ARGS, DEFAULT_CAMERA_ARGS, DEFAULT_COLOR_ARGS
from IPython import embed; 

# We'll use instance randomization so that entire geom groups are randomized together
macros.USING_INSTANCE_RANDOMIZATION = True

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = 'Lift'

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    options["robots"] = 'Jaco'

    # Choose controller
    controller_name = "OSC_POSE"

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        camera_names="agentview",
        render_camera="nearfrontview",
        control_freq=20,
        hard_reset=False,   # TODO: Not setting this flag to False brings up a segfault on macos or glfw error on linux
    )

    leave_out_color_geoms = ['cube', 'sphere', 'gripper', 'robot']
    use_color_geoms = []
    for g in env.sim.model.geom_names:
        include = True
        for lo in leave_out_color_geoms:
            if lo.lower() in g.lower():
                include = False
        if include:
            use_color_geoms.append(g)

    color_args = DEFAULT_COLOR_ARGS
    color_args['geom_names'] = use_color_geoms
    color_args['local_rgb_interpolation'] = .3
    color_args['local_material_interpolation'] = .3
    color_args['randomize_texture_images'] = True
    dynamics_args = DEFAULT_DYNAMICS_ARGS
    camera_args = DEFAULT_CAMERA_ARGS
    lighting_args = DEFAULT_LIGHTING_ARGS

    # Get action limits
    low, high = env.action_spec
    for re in range(20):
        env = GymImageDomainRandomizationWrapper(env, 
                randomize_dynamics=True, 
                dynamics_randomization_args=dynamics_args,
                randomize_camera=True, 
                camera_randomization_args=camera_args,
                randomize_color=True, 
                color_randomization_args=color_args,
                randomize_lighting=True, 
                lighting_randomization_args=lighting_args,
                randomize_on_reset=False, 
                randomize_every_n_steps=0)
        env.reset()
        for e in range(5):
            for i in range(40):
                action = np.random.uniform(low, high)
                obs, reward, done, _ = env.step(action)
                env.render()
            env.reset()
        #env.modders[0].change_default_texture('table_visual', np.random.randint(len(env.modders[0].textures)))
