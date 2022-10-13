#date: 2022-10-13T17:28:43Z
#url: https://api.github.com/gists/fe680ff96f83cc127bf385cd76b42a25
#owner: https://api.github.com/users/mateidanut

class Solution:
    def rob(self, nums: List[int]) -> int:
        robIncluding = []
        robIncluding.append(nums[0])
        
        if len(nums) > 1:
            robIncluding.append(nums[1])
        
        if len(nums) > 2:
            robIncluding.append(nums[2] + robIncluding[0])
        
        for i in range(3, len(nums)):
            robIncluding.append(nums[i] + max(robIncluding[i-2], robIncluding[i-3]))
        
        return max(robIncluding)