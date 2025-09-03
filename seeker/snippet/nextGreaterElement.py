#date: 2025-09-03T17:01:11Z
#url: https://api.github.com/gists/eb1630ac48f3253bd104d032a2bf345f
#owner: https://api.github.com/users/omeryanay1

from typing import List

def nextGreaterElement(nums: List[int]) -> List[int]:
    n = len(nums)
    res = [-1] * n
    stack = []  # indices of a decreasing stack

    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            res[stack.pop()] = x
        stack.append(i)
    
    return res