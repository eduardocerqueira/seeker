#date: 2025-09-08T17:11:56Z
#url: https://api.github.com/gists/4352a3c0be5a9a4e93f183bac12eac16
#owner: https://api.github.com/users/motyzk

class Solution:
    def twoSum(self, nums, target):
        complements = {}
        for i, n in enumerate(nums):
            if n in complements:
                return [complements[n], i]
            complements[target - n] = i
