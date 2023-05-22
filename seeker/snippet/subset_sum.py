#date: 2023-05-22T16:49:55Z
#url: https://api.github.com/gists/3b9a5d3d139c2ed987e0cc5188a9b838
#owner: https://api.github.com/users/Sinha-Ujjawal

from mknapsack import solve_subset_sum
import numpy as np


def subset_sum(*, weights: np.array, capacity: float) -> np.array:
    assert capacity > 0, f"Capacity must be positive!, given: {capacity}"
    
    n_weights = len(weights)
    
    if len(weights) < 2:
        return np.zeros(shape=n_weights, dtype=np.uint8)
    
    weights_to_consider_bool = (weights > 0) & (weights <= capacity)
    weights_to_consider = weights[weights_to_consider_bool]
    weights_sum = weights_to_consider.sum()
    
    if weights_sum == capacity:
        return weights_to_consider_bool.astype(np.uint8)
    
    if (len(weights_to_consider) < 2) or (weights_sum < capacity):
        abs_diff = np.abs(capacity - weights)
        diff1 = abs_diff.min()
        diff2 = capacity - weights_sum
        
        if diff1 <= diff2:
            ret = np.zeros(shape=n_weights, dtype=np.uint8)
            ret[np.where(abs_diff == diff1)] = 1
            return ret

        return weights_to_consider_bool.astype(np.uint8)
    
    ret = np.zeros(shape=n_weights, dtype=np.uint8)
    ret[weights_to_consider_bool] = solve_subset_sum(
        weights=weights_to_consider,
        capacity=capacity,
        method_kwargs={"check_inputs": False},
    )
    return ret