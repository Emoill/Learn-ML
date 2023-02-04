import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os

env_name = 'CartPole-v1'

env = gym.make(env_name)

ppo_path = os.path.join('Cartpole_model')
model = PPO.load(ppo_path, env) #load model    

#evaluate_policy(model, env, n_eval_episodes=10, render=True)

for episode in range(1, 11):
    score = 0
    obs = env.reset()
    done = False
    
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        
    print('Episode:', episode, 'Score:', score)
env.close()