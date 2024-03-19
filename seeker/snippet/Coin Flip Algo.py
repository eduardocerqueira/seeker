#date: 2024-03-19T17:01:27Z
#url: https://api.github.com/gists/b05762f25773d3540642344517238eac
#owner: https://api.github.com/users/srkim

import numpy as np

def simulate_coin_flips(n_flips=100, n_trials=10000):
    alice_wins = 0
    bob_wins = 0
    ties = 0
    
    for _ in range(n_trials):
        # Simulate the sequence of coin flips: 0 for Tails, 1 for Heads
        flips = np.random.randint(2, size=n_flips)
        
        # Find scores for Alice (HH) and Bob (HT)
        alice_score = sum(1 for i in range(len(flips) - 1) if flips[i] == 1 and flips[i+1] == 1)
        bob_score = sum(1 for i in range(len(flips) - 1) if flips[i] == 1 and flips[i+1] == 0)
        
        if alice_score > bob_score:
            alice_wins += 1
        elif bob_score > alice_score:
            bob_wins += 1
        else:
            ties += 1
            
    return alice_wins, bob_wins, ties

# Simulate the game for 10000 trials
alice_wins, bob_wins, ties = simulate_coin_flips()

alice_wins, bob_wins, ties
