
import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

env_name = 'CartPole-v1'

env = gym.make(env_name)

env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)

a,b,c = model.learn(total_timesteps=100)
print(a)

ppo_path = os.path.join('Cartpole_model')
model.save(ppo_path)


