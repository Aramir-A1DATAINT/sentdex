import gym
from gym import wrappers
from stable_baselines3 import PPO
import os
model_dir = "models/PPO"
logdir = 'logs'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

env = gym.make("LunarLander-v2")
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir) #tensorboard_log=logdir make log 
TIMESTEPS = 10000
for i in range(1, 30): 
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO") # model reset FALSE FOR every learning
    model.save(f"{model_dir}/{TIMESTEPS*i}")

# model learning episode 보기 
'''
model_dir = "models/PPO"
model_path = f"{model_dir}/180000.zip"

model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, state = model.predict(obs) # 여기 부분이 기존과 다름
        obs, reward, done, info = env.step(action)
        print(reward)
env.close()
''' 