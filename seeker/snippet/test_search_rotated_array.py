#date: 2024-12-23T16:48:35Z
#url: https://api.github.com/gists/ac181c04d83b532d7e75277f47538f7d
#owner: https://api.github.com/users/jymchng

import pytest
from search_rotated_array import search_rotated_array

@pytest.mark.parametrize("nums, target, expected", [
    # General cases
    ([4,5,6,7,0,1,2], 0, 4),
    ([4,5,6,7,0,1,2], 3, -1),
    ([1], 0, -1),
    
    # Edge cases
    ([1, 3, 5], 3, 1),
    ([2, 3, 4, 5, 6, 7, 8, 1], 8, 6),
    ([2, 3, 4, 5, 6, 7, 8, 1], 2, 0),
    ([7, 8, 1, 2, 3, 4, 5, 6], 3, 4),
    
    # Single element arrays
    ([1], 1, 0),
    ([1], 2, -1),
    
    # Two element arrays
    ([2, 1], 1, 1),
    ([2, 1], 2, 0),
    
    # Rotated arrays with large gaps
    ([30, 40, 50, 10, 20], 10, 3),
    ([30, 40, 50, 10, 20], 50, 2),
    
    # Searching for elements not in the array
    ([4,5,6,7,0,1,2], 8, -1),
    ([4,5,6,7,0,1,2], -1, -1),
    
    # Repeated element test for uniqueness check
    ([10, 20, 30, 40, 50, 60, 5], 60, 5),
    
    # Longer array
    ([15, 18, 2, 3, 6, 12], 15, 0),
    ([15, 18, 2, 3, 6, 12], 6, 4),
    
    # More complex rotations
    ([9,12,15,18,21,3,6], 21, 4),
    ([9,12,15,18,21,3,6], 6, 6),
])
def test_search_rotated_array(nums, target, expected):
    assert search_rotated_array(nums, target) == expected