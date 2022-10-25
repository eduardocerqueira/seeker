#date: 2022-10-25T17:49:12Z
#url: https://api.github.com/gists/f56c0f8e41393b41039f2b0f9f54b09a
#owner: https://api.github.com/users/keshavchand

import numpy as np

def SingleInput():
    #Assuming 3 input 
    inputs = [1,2,3]
    # Layer 1
    # 4 neurons
    weights = np.array([
        [0.20, 0.30, 0.40],
        [1.20, 1.30, 1.40],
        [2.20, 2.30, 2.40],
        [3.20, 3.30, 3.40],
        ])
    biases = np.array([10, 20, 30, 40])
    
    # A = weights * input + biases
    outputs = (weights @ inputs) + biases
    print(outputs)
    # [12. 28. 44. 60.]
    # print(12 == (0.20 * 1 + 0.30 * 2 + 0.40 * 3) + 10)


def MultipleInput():
    # Multiple Inputs
    # Each row is one input
    inputs = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9],
        ])
    # Layer 1
    # 4 neurons
    weights = np.array([
        [0.20, 0.30, 0.40],
        [1.20, 1.30, 1.40],
        [2.20, 2.30, 2.40],
        [3.20, 3.30, 3.40],
        ])
    biases = np.array([10, 20, 30, 40])
    
    # A = weights * input + biases
    outputs = (weights @ inputs.T).T + biases
    print(outputs)
    # [[ 12.   28.   44.   60. ]
    #  [ 14.7  39.7  64.7  89.7]
    #  [ 17.4  51.4  85.4 119.4]]
