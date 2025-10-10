#date: 2025-10-10T16:55:07Z
#url: https://api.github.com/gists/425dbe28b862bba0c4d66940e0a2ecc5
#owner: https://api.github.com/users/pepnova-9

"""
What I learned from this question.

- This is a good question to check the interviewee's coding ability.
  - Writing up code to search all the patterns(brute force way) is sometimes hard.
  - In this problem, we have to consider all the possible order to perform arithmetic operations.
    - ex: (((a + b) * c) / d). This is the simplest because we perform the operation from left to right.
    - but, we also need to consider, for example, a * (b + c) / d this order of operation.
- Needs to consider rounding error.
  - Most likely, we will be asked about rounding error. 
  - Probably, be asked about when and why does rounding error happen?
"""
from itertools import permutations
class Solution:
    def judgePoint24(self, cards: List[int]) -> bool:
        target_num = 24
        EPSILON = 1e-6
        
        def solve(nums):
            if len(nums) == 1:
                return abs(nums[0] - target_num) < EPSILON

            for i in range(len(nums)):
                for j in range(len(nums)):
                    if i == j:
                        continue
                    
                    a, b = nums[i], nums[j]                    
                    remaining_nums = [nums[k] for k in range(len(nums)) if k not in (i, j)]

                    if solve(remaining_nums + [a + b]):
                        return True

                    if solve(remaining_nums + [a - b]):
                        return True

                    if solve(remaining_nums + [a * b]):
                        return True

                    if b != 0 and solve(remaining_nums + [a / b]):
                        return True
            return False

        return solve([float(num) for num in cards])
