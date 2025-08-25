#date: 2025-08-25T17:08:20Z
#url: https://api.github.com/gists/1010989d4e976fe26d1a5d4b2c533cff
#owner: https://api.github.com/users/ayhanasghari

# train_rl.py

from stable_baselines3 import DQN
from tabriz_traffic_env import TabrizTrafficEnv

env = TabrizTrafficEnv(graph_path="tabriz_iran.graphml")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("dqn_tabriz_model")
