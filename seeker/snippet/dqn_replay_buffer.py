#date: 2023-01-05T16:40:08Z
#url: https://api.github.com/gists/54699cb404055cb3fa9f7dc9919de74e
#owner: https://api.github.com/users/JavierMtz5

from collections import deque
import numpy as np
import random

replay_buffer = deque(maxlen=100000)

def save_experience(state, action, reward, next_state, terminal):

    # Save the given experience as a (s, a, r, s', terminal) tuple
    replay_buffer.append((state, action, reward, next_state, terminal))

def sample_experience_batch(batch_size):

    # Sample {batchsize} experiences from the ReplayBuffer
    exp_batch = random.sample(replay_buffer, batch_size)

    # Create an array with the {batchsize} elements for s, a, r, s' and terminal information
    state_batch = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, state_size)
    action_batch = np.array([batch[1] for batch in exp_batch])
    reward_batch = [batch[2] for batch in exp_batch]
    next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, state_size)
    terminal_batch = [batch[4] for batch in exp_batch]

    # Return a tuple, where each item corresponds to each array/batch created above
    return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch