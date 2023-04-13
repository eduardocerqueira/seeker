#date: 2023-04-13T17:01:59Z
#url: https://api.github.com/gists/f6a67ea2068aee5abf21c9172d8fb0b7
#owner: https://api.github.com/users/Artur-Galstyan

import jax 
import jax.numpy as jnp

@jax.jit
def get_discounted_rewards(rewards: jnp.ndarray, gamma=0.99) -> float:
    """Calculate the discounted rewards for a given set of rewards.
    Args:
        rewards: The rewards to calculate the discounted rewards for.
        gamma: The discount factor.

    Returns:
        The discounted rewards.
    """
    def body_fn(i: int, val: float):
        return val + (gamma ** i) * rewards[i]
    
    discounted_rewards = jnp.zeros(())
    num_rewards = len(rewards)
    discounted_rewards = jax.lax.fori_loop(0, num_rewards, body_fn, discounted_rewards)
    
    return discounted_rewards

@jax.jit
def get_total_discounted_rewards(rewards: jnp.array, gamma=0.99) -> jnp.array:
    """Calculate the total discounted rewards for a given set of rewards.
    Args:
        rewards: The rewards to calculate the total discounted rewards for.
        gamma: The discount factor.

    Returns:
        The total discounted rewards.
    """
    total_discounted_rewards = jnp.zeros(shape=rewards.shape)
    for i in range(len(rewards)):
        current_slice = rewards[i:]
        discounted_reward = get_discounted_rewards(current_slice, gamma)
        total_discounted_rewards = total_discounted_rewards.at[i].set(discounted_reward)
    return total_discounted_rewards.reshape(-1, 1)

