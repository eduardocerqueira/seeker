#date: 2023-06-20T16:43:56Z
#url: https://api.github.com/gists/8f9c02c30ea87062cd548b816216aaa3
#owner: https://api.github.com/users/lucasnfe

import gym
import numpy as np

# Initialize the CartPole environment
env = gym.make('CartPole-v1')

# Set hyperparameters
num_episodes = 5000
max_steps_per_episode = 100
alpha = 0.15
gamma = 0.995
epsilon = 0.2

# Discretize the state space
num_bins = 30
state_bins = [
    np.linspace(env.observation_space.low[i], env.observation_space.high[i], num_bins - 1) for i in range(4)
]

# Initialize the Q-table
action_space_size = env.action_space.n
state_space_size = (num_bins,) * env.observation_space.shape[0]

q_table = np.zeros((state_space_size + (action_space_size,)))

# Convert continuous state to discrete indices
def discretize_state(state):
    indices = []
    for i in range(len(state)):
        indices.append(np.digitize(state[i], state_bins[i]))
    return tuple(indices)

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    state = discretize_state(state)
    
    total_reward = 0

    done = False
    for step in range(max_steps_per_episode):
        # Render the environment
        env.render()  

        # Exploration-exploitation trade-off: epsilon-greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        # Perform the chosen action
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Update Q-table
        q_table[state + (action,)] = (1 - alpha) * q_table[state + (action,)] + \
                                     alpha * (reward + gamma * np.max(q_table[next_state]))

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}: Total reward = {total_reward}")

# Close the environment
env.close()
