from tabnanny import verbose
import gym
from gym import wrappers
from stable_baselines3 import A2C

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
env = gym.make("LunarLander-v2")

env.reset()

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        print(reward)
env.close()