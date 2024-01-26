#date: 2024-01-26T17:08:28Z
#url: https://api.github.com/gists/80fc53b0b79f258bd97d98b02cc8fe61
#owner: https://api.github.com/users/perlucas

from envs_.trading_env import TradingEnv
from airtos.envs import MacdEnv
from utils import load_dataset
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
import matplotlib.pyplot as plt

# load dataset and create new environment instance
# use MACD TI, use window of 10 ticks
df = load_dataset('./KO.csv')
env = MacdEnv(df=df, window_size=10, frame_bound=(10, 1000))

# create tf environment instance and random agent (takes actions randomly)
eval_env = tf_py_environment.TFPyEnvironment(env)
random_policy = random_tf_policy.RandomTFPolicy(eval_env.time_step_spec(), eval_env.action_spec())

# helper function to render the environment
def render_policy_eval(policy, filename):
    time_step = eval_env.reset()
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
    env.render('human')
    
# run and render the environment
plt.figure(figsize=(15, 8)) # set up python plt output screen
render_policy_eval(random_policy, "random_agent")