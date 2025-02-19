#date: 2025-02-19T16:46:18Z
#url: https://api.github.com/gists/ff56a38293867ea8ef15ae375fde3d09
#owner: https://api.github.com/users/qirh

"""
Similar to LC 378 -- The heap solution is more efficient
You have m arrays of sorted integers. Find the kth smallest value of all the values. The sum of the arrays' lengths is n.

Example:
Input Arrays: [
    [1, 3, 5, 7],
    [2, 4, 6],
    [8, 9]
]
(k=5, m=3, n=9)

Output: 5
"""

from collections import deque

def get_kth_smallest_value(lists, k, n):
    counter = 1
    curr_value = None

    queues = [deque(l) for l in lists]

    while counter <= k:
        smallest_el = None
        smallest_q = None

        for q in queues:
            if len(q) > 0 and (smallest_el is None or q[0] < smallest_el):
                smallest_el = q[0]
                smallest_q = q
        curr_value = smallest_el
        smallest_q.popleft()
        counter += 1


    return curr_value
