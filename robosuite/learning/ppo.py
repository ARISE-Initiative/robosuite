import sys

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os
import numpy as np

import robosuite
from robosuite.wrappers import TeleopWrapper, GymWrapper, IKWrapper

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO2
from stable_baselines.bench import Monitor

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    global n_steps, best_mean_reward
    if (n_steps + 1) % 75 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True

def main():
# Create log dir
    log_dir = "./checkpoints/reach/lift_4d/"
    os.makedirs(log_dir, exist_ok=True)

    num_stack = 3
    num_env = 2
    render = False
    print('Config for ' + log_dir + ':')
    print('num_stack:', num_stack)
    print('num_env:', num_env)
    print('render:', render)

    env = GymWrapper(IKWrapper(robosuite.make("SawyerLift", has_renderer=render, has_offscreen_renderer=False, use_camera_obs=False, reward_shaping=True)), stack=num_stack)
    env.metadata = {'render.modes': ['human']}
    env.reward_range = None
    env.spec = None
    env = Monitor(env, log_dir, allow_early_resets=True)
    #env = VecFrameStack(SubprocVecEnv([lambda: env] * num_env, 'fork'), num_stack)
    env = VecFrameStack(DummyVecEnv([lambda: env] * num_env), num_stack)

    try:
        print('Trying existing model...')
        model = PPO2.load(log_dir + 'best_model.pkl')
        model.set_env(env)
    except:
        print('No existing model found. Training new one.')
        model = PPO2(MlpPolicy, env, verbose=1)

    model.learn(total_timesteps=int(1e4), callback=callback)

    if render:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            env._get_target_envs([0])[0].render()
            if done[0]:
                obs = env.reset()

if __name__ == '__main__':
    main()
