#date: 2023-06-20T16:42:37Z
#url: https://api.github.com/gists/b800a403833694953b1e3b507666bce9
#owner: https://api.github.com/users/lucasnfe

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, n):
        self.n = n
        self.q_true = np.random.randn(n) # True action values (unknown)
        self.q_estimate = np.zeros(n)    # Estimated action values
        self.action_count = np.zeros(n)  # Number of times each action was taken

    def get_reward(self, action):
        # Generate reward with a mean equal to the true action value and unit variance
        return np.random.normal(self.q_true[action], 1)

    def epsilon_greedy(self, epsilon):
        if np.random.random() < epsilon:
            # Exploration: choose a random action
            action = np.random.randint(self.n)
        else:
            # Exploitation: choose the action with the highest estimated value
            action = np.argmax(self.q_estimate)

        reward = self.get_reward(action)
        self.action_count[action] += 1

        # Update the estimated action value using sample-average method
        alpha = 1.0 / self.action_count[action]
        self.q_estimate[action] += alpha * (reward - self.q_estimate[action])

        return reward

    def ucb(self, c):
        ucb_estimates = self.q_estimate + c * np.sqrt(np.log(np.sum(self.action_count)) / (self.action_count + 1e-5))
        action = np.argmax(ucb_estimates)
        reward = self.get_reward(action)
        self.action_count[action] += 1

        # Update the estimated action value using sample-average method
        alpha = 1.0 / self.action_count[action]
        self.q_estimate[action] += alpha * (reward - self.q_estimate[action])
        
        return reward

# Parameters
num_bandits = 10  # Number of bandit arms
num_steps = 1000  # Number of time steps
epsilons = [0, 0.1, 0.01]  # Epsilon values for epsilon-greedy algorithm
c = 2  # Exploration parameter for UCB algorithm
num_experiments = 2000  # Number of experiments to run

rewards_per_step = np.zeros((len(epsilons) + 1, num_steps))  # +1 for UCB algorithm

# Run experiments for each epsilon value
for i, epsilon in enumerate(epsilons):
    for _ in range(num_experiments):
        bandit = Bandit(num_bandits)
        rewards = np.zeros(num_steps)
        for step in range(num_steps):
            reward = bandit.epsilon_greedy(epsilon)
            rewards[step] = reward
        rewards_per_step[i] += rewards

# Run experiments for UCB algorithm
for _ in range(num_experiments):
    bandit = Bandit(num_bandits)
    rewards = np.zeros(num_steps)
    for step in range(num_steps):
        reward = bandit.ucb(c)
        rewards[step] = reward
    rewards_per_step[-1] += rewards

# Calculate average reward per step for each algorithm
rewards_per_step /= num_experiments

# Plot average reward per step for each algorithm
labels = epsilons + ["UCB"]
for i, label in enumerate(labels):
    plt.plot(rewards_per_step[i], label=label)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Comparison of Epsilon-Greedy and UCB Algorithms")
plt.legend()
plt.show()
