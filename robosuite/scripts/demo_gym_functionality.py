"""
This script shows how to adapt an environment to be compatible
with the OpenAI gym API. This is extremely useful when using
learning pipelines that require supporting these APIs.

For instance, this can be used with OpenAI Baselines
(https://github.com/openai/baselines) to train agents
with RL.



We base this script off of some code snippets found
in the "Getting Started with Gym" section of the OpenAI 
gym documentation.

The following snippet was used to show how to list all environments.
    from gym import envs
    print(envs.registry.all())

The following snippet was used to demo basic functionality.

    import gym
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

"""


"""
The following snippet was used to show how to list all environments.
    from gym import envs
    print(envs.registry.all())

We demonstrate equivalent functionality below.
"""

import robosuite as suite

# get the list of all environments
envs = sorted(suite.ALL_ENVIRONMENTS)
print("Welcome to Surreal Robotics Suite v{}!".format(suite.__version__))
print(suite.__logo__)
print("Here is a list of environments in the suite:\n")

for k, env in enumerate(envs):
    print("[{}] {}".format(k, env))
print()

"""
The following snippet was used to demo basic functionality.

    import gym
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

We demonstrate equivalent functionality below.
"""

from robosuite.wrappers import GymWrapper

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "Lift",
            robots="Sawyer",        # use Sawyer robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=100,  # control should happen fast enough so that simulation looks smooth
        )
    )

    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
