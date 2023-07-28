#date: 2023-07-28T16:37:34Z
#url: https://api.github.com/gists/14187a93b884eb84e910b3fca2ac558e
#owner: https://api.github.com/users/abhiphile

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d={}
        for i in range(len(nums)):
            if target-nums[i] in d:
                return [d[target-nums[i]],i]
            d[nums[i]]=i
        return []