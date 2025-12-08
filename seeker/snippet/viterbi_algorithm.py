#date: 2025-12-08T16:59:34Z
#url: https://api.github.com/gists/a0c14028ac054256184c7a635b316bb7
#owner: https://api.github.com/users/dopolian

import numpy as np

def viterbi_general_bags(marble_sequence, bag_configs, transition_matrix, initial_probs=None):
    """
    Viterbi algorithm: finds most likely sequence of hidden states (bags)
    given observed events (marble colors).
    
    Args:
        marble_sequence: List of observed colors, e.g., ['black', 'red', 'black']
        bag_configs: Marble counts per bag, e.g., {'A': {'red': 4, 'black': 1}}
        transition_matrix: Transition probabilities, e.g., {'A': {'A': 0.8, 'B': 0.2}}
        initial_probs: Starting probabilities (uses steady-state if None)
    
    Returns:
        (bag_sequence, probability): Most likely path and its probability
    """
    states = list(bag_configs.keys())
    
    # Convert marble counts to probabilities: P(color | bag)
    emit_prob = {}
    # emission probabilities are derived from bag configurations of marbles (ex : bag A has 4 red and 1 black marble)
    for bag, marbles in bag_configs.items():
        total = sum(marbles.values())
        emit_prob[bag] = {color: count/total for color, count in marbles.items()}
    
    # Use steady-state if no initial probabilities provided
    if initial_probs is None:
        initial_probs = compute_steady_state(transition_matrix, states)
    
    n = len(marble_sequence)
    
    # V[t][state] = max probability of path ending at state at time t
    V = [{} for _ in range(n)]
    # path[state] = sequence of states in most likely path to this state
    path = {}
    
    # Step 1: Initialize with first observation
    for state in states:
        V[0][state] = initial_probs[state] * emit_prob[state][marble_sequence[0]]
        path[state] = [state]
    
    # Step 2: For each subsequent observation, find best previous state
    for t in range(1, n):
        new_path = {}
        
        for curr_state in states:
            max_prob = 0
            best_prev = None
            
            # Find which previous state maximizes: 
            # P(prev_path) * P(transition) * P(observation|curr_state)
            for prev_state in states:
                prob = (V[t-1][prev_state] *                           # Max prob to prev_state
                       transition_matrix[prev_state][curr_state] *    # Transition prob
                       emit_prob[curr_state][marble_sequence[t]])     # Emission prob
                
                if prob > max_prob:
                    max_prob = prob
                    best_prev = prev_state
            
            # Store best probability and extend best path
            V[t][curr_state] = max_prob
            new_path[curr_state] = path[best_prev] + [curr_state]
        
        path = new_path
    
    # Step 3: Find state with highest probability at final time
    best_final = max(V[n-1], key=V[n-1].get)
    final_prob = V[n-1][best_final]
    
    return path[best_final], final_prob


def compute_steady_state(trans_matrix, states):
    """
    Solves wT = w to find steady-state distribution.
    Long-run probability of being in each state.

    Args:
        trans_matrix: Transition probability matrix as a dict of dicts
        states: List of states
    Returns:
        steady_state: Dict of steady-state probabilities
    """
    n = len(states)
    
    # Convert dict to numpy matrix
    T = np.zeros((n, n))
    for i, state_from in enumerate(states):
        for j, state_to in enumerate(states):
            T[i][j] = trans_matrix[state_from][state_to]
    
    # Solve: (T^T - I)w = 0 with constraint sum(w) = 1
    A = np.vstack([T.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1
    
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return {state: w[i] for i, state in enumerate(states)}


### TEST CASES ###

# Example: 3-bag problem

# Marble Counts
bags_3 = {
    'A': {'red': 8, 'black': 2},
    'B': {'red': 5, 'black': 5},
    'C': {'red': 3, 'black': 7}
}

# Transition Matrix
trans_3 = {
    'A': {'A': 0.4, 'B': 0.4, 'C': 0.2},
    'B': {'A': 0.2, 'B': 0.4, 'C': 0.4},
    'C': {'A': 0.5, 'B': 0.1, 'C': 0.4}
}

# Sequence of 25 marbles drawn
seq_3 = ['red', 'red', 'black', 'black', 'red', 'black', 'red', 'red', 'black', 'red', 'black', 'black', 'red', 'red', 'black', 'red', 'black', 'red', 'red', 'black', 'black', 'black', 'red', 'red', 'black']

result_3, prob_3 = viterbi_general_bags(seq_3, bags_3, trans_3)

print(f"Sequence: {seq_3}")
print(f"Bags: {result_3}")
print(f"Probability: {prob_3:.8f}")

# Example: Skewed 2-bag problem (edge case)

# Marble Counts
bags_skewed = {
    'A': {'red': 9, 'black': 1},
    'B': {'red': 1, 'black': 9}
}

# Transition Matrix
trans_skewed = {
    'A': {'A': 0.5, 'B': 0.5},
    'B': {'A': 0.5, 'B': 0.5}
}

# Sequence of 10 marbles drawn
seq_skewed = ['red', 'red', 'red', 'black', 'red', 'red', 'red', 'red', 'black', 'red']
result_skewed, prob_skewed = viterbi_general_bags(seq_skewed, bags_skewed, trans_skewed)
# Red clusters map to bag A, black marbles to bag B
print(f"\nSequence: {seq_skewed}")
print(f"Bags: {result_skewed}")
print(f"Probability: {prob_skewed:.8f}")