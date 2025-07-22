#date: 2025-07-22T17:11:59Z
#url: https://api.github.com/gists/9f7f169e58fb47fdf9b36fcdfc1bb187
#owner: https://api.github.com/users/varnie

"""
Missing number


You are given an array nums containing n different numbers in the range [0, n]. 
You need to return the only number from the range that is missing from the array.

Example 1:

Input: nums = [3,0,1]

Output: 2

Explanation: n = 3, since there are 3 numbers in the array, therefore all the numbers are in the range [0,3]. 
The number 2 is not in this range, since it does not appear in nums.

Example 2:

Input: nums = [0,1]

Output: 2

Explanation: n = 2, since there are 2 numbers in the array, therefore all the numbers are in the range [0,2]. 
The number 2 is not in this range, since it does not appear in nums.

Example 3:

Input: nums = [9,6,4,2,3,5,7,0,1]

Output: 8

Explanation: n = 9, since there are 9 numbers in the array, therefore all numbers are in the range [0,9]. The number 8 is not in this range, since it does not occur in nums.
"""

class Solution:
    def missing_number(self, nums: list[int]) -> int:
        nums.sort()

        prev_num = None
        for i, val in enumerate(nums):
            is_first_go = i == 0
            if is_first_go:
                prev_num = val
                continue
            
            if prev_num + 1 != val:
                return prev_num+1
            prev_num = val
        else:
            return len(nums)
            