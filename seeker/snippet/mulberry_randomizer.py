#date: 2025-05-29T17:10:03Z
#url: https://api.github.com/gists/b6f21c0ce6f3e83bd4f9b3d86e6a7556
#owner: https://api.github.com/users/EncodeTheCode

import time
import struct
import secrets

# Mulberry32 PRNG implementation in Python
def create_random(seed):
    def rng():
        nonlocal seed
        seed = seed & 0xFFFFFFFF  # Ensure 32-bit
        seed = (seed + 0x6D2B79F5) & 0xFFFFFFFF
        t = (seed ^ (seed >> 15)) * (1 | seed)
        t = (t + ((t ^ (t >> 7)) * (61 | t))) ^ t
        t = t ^ (t >> 14)
        return (t & 0xFFFFFFFF) / 4294967296
    return rng

# Generate secure seed
try:
    seed_bytes = "**********"
    seed = struct.unpack("I", seed_bytes)[0]
except Exception:
    seed = int(time.time() * 1000) & 0xFFFFFFFF

# Create the RNG
random = create_random(seed)

# Luck array
luck = [0.25, 0.1, 0.5, 1.0, 0.75, 0.65]

# Select a random probability from the luck array
def get_random_prob():
    index = int(random() * len(luck))
    return luck[index]

# Final chance function using dynamic or provided probability
def random_chance(prob=None):
    p = prob if isinstance(prob, (int, float)) else get_random_prob()
    return random() < p

# Example use:
from_ = {"armies": 10}
to = {"armies": 5}

if random_chance():
    from_["armies"] -= 1
    to["armies"] -= 0
else:
    from_["armies"] -= 1
    to["armies"] -= 1
es"] -= 1
