#date: 2022-05-20T17:09:00Z
#url: https://api.github.com/gists/228fc5610ec767bb88577381a9b73c44
#owner: https://api.github.com/users/Markus28

import gym
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def simulate_vector(step_samples, n_envs, n_samples):
    total = 0
    for _ in range(n_samples):
        execution_time = np.max(np.random.choice(step_samples, n_envs, replace=True))
        total += execution_time

    return total / n_samples

env_names = ["Ant-v3", "LunarLander-v2", "CartPole-v1"]
n_samples_step = 25_000
n_bins = n_samples_step // 400
n_envs = 2**np.arange(12)
fig, ax = plt.subplots(nrows=2, ncols=len(env_names))

for i, env_name in enumerate(env_names):
    env = gym.vector.make(env_name)
    env.reset()

    auto_reset_times = []
    only_step_times = []

    print(f"Sampling {env_name}...")
    for _ in tqdm(range(n_samples_step)):
        t0 = time.time()
        _, _, _, info = env.step(env.action_space.sample())
        t1 = time.time()
        dt = 1000*(t1-t0)
        auto_reset = "terminal_observation" in info[0]
        if auto_reset:
            auto_reset_times.append(dt)
        else:
            only_step_times.append(dt)

        
    ax[0, i].set_title(f"{env_name}")
    ax[0, i].set_xlabel("Time [ms]")
    ax[0, i].set_ylabel("Density")
    ax[0, i].hist([only_step_times, auto_reset_times], n_bins, label=["Only Step", "With Reset"], density=True, stacked=True)
    ax[0, i].legend()


    step_samples = np.array(only_step_times + auto_reset_times)

    execution_times = []

    for k in n_envs:
        execution_times.append(simulate_vector(step_samples, k, 15000))


    ax[1, i].scatter(n_envs, execution_times)
    ax[1, i].plot(n_envs, execution_times)
    ax[1, i].set_xlabel("# Sub-Environments")
    ax[1, i].set_xscale("log")
    ax[1, i].set_ylabel("Expected Time [ms]")
    ax[1, i].set_ylim([0, 1.1*max(execution_times)])
    ax[1, i].grid()

fig.tight_layout()
plt.show()
