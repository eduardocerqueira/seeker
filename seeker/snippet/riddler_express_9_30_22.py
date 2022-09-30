#date: 2022-09-30T17:03:03Z
#url: https://api.github.com/gists/e86a1c33f29a894aa639211e0de235c4
#owner: https://api.github.com/users/bharathjaladi

import numpy as np
from collections import defaultdict

# Encoding of each digit on a 7-segment display.
encoding = np.array([[1,1,1,1,1,1,0], [0,1,1,0,0,0,0], [1,1,0,1,1,0,1], [1,1,1,1,0,0,1], [0,1,1,0,0,1,1], [1,0,1,1,0,1,1], 
                     [1,0,1,1,1,1,1], [1,1,1,0,0,0,0], [1,1,1,1,1,1,1], [1,1,1,1,0,1,1]])

# Patterns that would be created by the fading of the two-digit passcode. (By summing the encodings of each digit, we arrive at an 
# encoding for the fading, where a 1 represents a segment that is faded, a 2 represents a segment that is twice as faded, and a 0 
# represents a segment that is not faded.)
patterns = defaultdict(list)
for i, first in enumerate(encoding):
    for j, second in enumerate(encoding):
        patterns[tuple(first + second)].append(f'{i}{j}')

# Find patterns with more than 2 possible passcodes. (Any two digits in either order will produce the same fading pattern, so we want 
# at least two different sets of digits that produce the same fading pattern since the safe cannot be opened with confidence.)
for pattern, passcodes in patterns.items():
    if len(passcodes) > 2:
        print(passcodes)

# Output: ['58', '69', '85', '96']