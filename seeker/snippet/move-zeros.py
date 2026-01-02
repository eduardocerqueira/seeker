#date: 2026-01-02T17:09:54Z
#url: https://api.github.com/gists/796449ce19abb7e169733def9cf583d4
#owner: https://api.github.com/users/primlock

"""
This is a solution for Leetcode #283 - Move Zeros

https://leetcode.com/problems/move-zeroes/
"""

from typing import List

class Solution:
    def MoveZeroes(self, nums: List[int]) -> None:
        left, right = 0, 0

        # keep going until there are no more numbers to swap with
        while right != (len(nums) - 1):
            if left < len(nums) and nums[left] == 0:
                # advance right until its at the next non-zero
                while right != (len(nums) - 1) and nums[right] == 0:
                    right += 1

                # do the swap
                tmp = nums[left]
                nums[left] = nums[right]
                nums[right] = tmp

                # only move the left
                left += 1
            else:
                # move the left and the right
                left += 1
                right += 1

        print(nums)

if __name__ == "__main__":
    solution = Solution()

    # start with a zero
    nums = [0,1,0,3,12] # [1,3,12,0,0]
    # nums = [0,0,1] # [1,0,0]
    # nums = [0] # [0]

    # start with non-zero
    # nums = [1,0] # [1,0]
    # nums = [1,0,1] # [1,1,0]

    solution.MoveZeroes(nums)
