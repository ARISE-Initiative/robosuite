import sys

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os
import numpy as np

import robosuite
from robosuite.wrappers import TeleopWrapper, GymWrapper, IKWrapper

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
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

# Create log dir
log_dir = "./checkpoints/reach/lift_4d"
os.makedirs(log_dir, exist_ok=True)

env = GymWrapper(IKWrapper(robosuite.make("SawyerLift", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, reward_shaping=True)))

env.metadata = {'render.modes': ['human']}
env.reward_range = None
env.spec = None
env = Monitor(env, log_dir, allow_early_resets=True) #env = Monitor(env, None)
env = DummyVecEnv([lambda: env] * 8)


print('Using existing model')
model = PPO2.load('4d_low_dim_4_envbest_model')
model.set_env(env)
#try:
#    print('Using existing model')
#    model = PPO2.load('4d_low_dim_4_envbest_model')
#    model.set_env(env)
#except:
#    print('Training new model')
#    model = PPO2(MlpPolicy, env, verbose=1, nminibatches=4)

model.learn(total_timesteps=int(1e1), callback=callback)

env = GymWrapper(IKWrapper(robosuite.make("SawyerLift", has_renderer=True, has_offscreen_renderer=False, use_camera_obs=False, reward_shaping=True, camera_name='agentview')))
obs = env.reset()

while True:
    obs = np.tile(obs, (2, 1))
    action, _states = model.predict(np.squeeze(obs))
    obs, rewards, done, info = env.step(action[0])
    env.render()
    if done:
        obs = env.reset()

