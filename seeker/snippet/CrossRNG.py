#date: 2025-04-21T17:09:23Z
#url: https://api.github.com/gists/c166c3d904d5c9120723f794916e6149
#owner: https://api.github.com/users/EmilySamantha80

"""
A simple pseudorandom number generator that produces identical sequences in Python and JavaScript.
Uses a Linear Congruential Generator algorithm with carefully chosen constants.
"""

import time

class CrossRNG:
    """
    A cross-language compatible pseudorandom number generator.
    """
    
    def __init__(self, seed=None):
        """
        Creates a new random number generator
        
        Args:
            seed: The seed value (default: current timestamp)
        """
        # Constants for the LCG algorithm
        self.m = 2147483647    # 2^31 - 1 (a prime number)
        self.a = 48271         # Recommended multiplier for this modulus
        self.c = 0             # Using a "multiplicative congruential generator" variant
        
        # Set initial state using seed
        if seed is None:
            self.state = int(time.time() * 1000) % self.m
        else:
            self.state = seed % self.m
            
        if self.state <= 0:
            self.state = 1  # Ensure positive seed
    
    def next_int(self):
        """
        Get the next random integer in the sequence (0 to m-1)
        
        Returns:
            A random integer
        """
        self.state = (self.a * self.state + self.c) % self.m
        return self.state
    
    def random(self):
        """
        Get the next random float in the range [0, 1)
        
        Returns:
            A random float between 0 (inclusive) and 1 (exclusive)
        """
        return self.next_int() / self.m
    
    def rand_int(self, min_val, max_val):
        """
        Get a random integer between min and max (inclusive)
        
        Args:
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            A random integer between min and max
        """
        min_val = int(min_val)
        max_val = int(max_val)
        return int(self.random() * (max_val - min_val + 1)) + min_val
    
    def seed(self, seed):
        """
        Set a new seed for the generator
        
        Args:
            seed: The new seed value
        """
        self.state = seed % self.m
        if self.state <= 0:
            self.state = 1  # Ensure positive seed


# Example usage
if __name__ == "__main__":
    # Create RNG with seed 12345
    rng = CrossRNG(12345)
    
    # Generate some random numbers
    print("First 5 random numbers with seed 12345:")
    for i in range(5):
        print(rng.random())