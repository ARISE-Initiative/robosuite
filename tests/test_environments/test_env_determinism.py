"""
Test all environments with random policies.

This runs some basic sanity checks on the environment, namely, checking that:
    - proprio-state exists in the obs, and is a flat array
    - agentview_image exists and is of the correct shape
    - no object-obs in state, because we are only using image observations

Obviously, if an environment crashes during runtime, that is considered a failure as well.
"""
import numpy as np

import robosuite as suite

def qpos_idx_to_joint_name(sim, idx):
    
    model = sim.model
    for name in model.joint_names:
        addr = model.get_joint_qpos_addr(name)
        if addr == idx:
            return name
        
    return None
 

def test_environment_determinism():

    envs = sorted(suite.ALL_ENVIRONMENTS)
    for env_name in envs:
        # use a new seed every time!
        seed = np.random.randint(0, 10000)
        # Create config dict
        env_config = {"env_name": env_name}
        for robot_name in ("Panda", "Sawyer", "Baxter", "GR1"):
            # create an environment for learning on pixels
            config = None
            if "TwoArm" in env_name:
                if robot_name == "GR1":
                    continue
                if robot_name == "Baxter":
                    robots = robot_name
                    config = "bimanual"
                else:
                    robots = [robot_name, robot_name]
                    config = "opposed"
                # compile configuration specs
                env_config["robots"] = robots
                env_config["env_configuration"] = config
            elif "Humanoid" in env_name:
                if robot_name != "GR1":
                    continue
                env_config["robots"] = robot_name
            else:
                if robot_name == "Baxter" or robot_name == "GR1":
                    continue
                env_config["robots"] = robot_name

            # Notify user of which test we are currently on
            print("Testing envs: {} with robots {} with config {}...".format(env_name, env_config["robots"], config))

            # Create environment
            env1 = suite.make(
                **env_config,
                has_renderer=False,  # no on-screen renderer
                has_offscreen_renderer=True,  # off-screen renderer is required for camera observations
                ignore_done=True,  # (optional) never terminates episode
                use_camera_obs=True,  # use camera observations
                camera_heights=84,  # set camera height
                camera_widths=84,  # set camera width
                camera_names="agentview",  # use "agentview" camera
                use_object_obs=False,  # no object feature when training on pixels
                reward_shaping=True,  # (optional) using a shaping reward
                seed=seed,  # set seed for reproducibility
            )
            env2 = suite.make(
                **env_config,
                has_renderer=False,  # no on-screen renderer
                has_offscreen_renderer=True,  # off-screen renderer is required for camera observations
                ignore_done=True,  # (optional) never terminates episode
                use_camera_obs=True,  # use camera observations
                camera_heights=84,  # set camera height
                camera_widths=84,  # set camera width
                camera_names="agentview",  # use "agentview" camera
                use_object_obs=False,  # no object feature when training on pixels
                reward_shaping=True,  # (optional) using a shaping reward
                seed=seed,  # set seed for reproducibility
            )

            obs = env1.reset()
            obs2 = env2.reset()


            env1_xml = env1.sim.model.get_xml()
            env2_xml = env2.sim.model.get_xml()
            env1_state = env1.sim.get_state().flatten()
            env2_state = env2.sim.get_state().flatten()
            
            for st_idx in range(env1_state.shape[0]):
                if env1_state[st_idx] != env2_state[st_idx]:
                    joint_name = qpos_idx_to_joint_name(env1.sim, st_idx)
                    if joint_name is not None:
                        print(f"Joint {joint_name} at index {st_idx} is different between env1 and env2.")
                    else:
                        print(f"State at index {st_idx} is different between env1 and env2.")

            if  env1_xml != env2_xml:
                print("Environment XMLs are not the same! xml1: {}, xml2: {}".format(env1_xml, env2_xml))
                assert False
            assert np.allclose(env1_state, env2_state), "Environment states are not the same!"
            env1.close()
            env2.close()

            

    # Tests passed!
    print("All environment tests passed successfully!")


if __name__ == "__main__":

    test_environment_determinism()
