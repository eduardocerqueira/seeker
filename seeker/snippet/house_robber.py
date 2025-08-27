#date: 2025-08-27T17:11:32Z
#url: https://api.github.com/gists/e2643354f48f4494519e25a6eeffdf24
#owner: https://api.github.com/users/motyzk

# https://leetcode.com/problems/house-robber/


# linear time/space
class Solution:
    def rob(self, nums) -> int:
        if not nums:
            return 0
        if len(nums) < 3:
            return max(nums)
        memo = [nums[0], max(nums[0], nums[1]),
                max(nums[1], nums[0] + nums[2])]
        for i in range(3, len(nums)):
            memo.append(max(nums[i] + memo[i-2], memo[i-1]))
        return memo[-1]


# not efficient. fix
class Solution:
    def __init__(self):
        self.cache = {}
    def rob(self, nums):
        nums = tuple(nums)
        def rec(nums):
            if not nums:
                return 0
            if len(nums) <= 2:
                return max(nums)
            if nums not in self.cache:
                self.cache[nums] = nums[0] + max(rec(nums[2:]), rec(nums[3:]))
            return self.cache[nums]
        return max(rec(nums), rec(nums[1:]))


nums = [2,3,2]
output = 4
s = Solution()
assert s.rob(nums) == output
nums = [1,2,3,1]
output = 4
s = Solution()
assert s.rob(nums) == output
nums = [2,7,9,3,1]
output = 12
s = Solution()
assert s.rob(nums) == output
