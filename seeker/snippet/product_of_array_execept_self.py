#date: 2023-07-28T16:52:10Z
#url: https://api.github.com/gists/9286b9d54537f7c7b134abaccac8d038
#owner: https://api.github.com/users/abhiphile

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        length=len(nums)
        sol=[1]*length
        pre = 1
        post = 1
        for i in range(length):
            sol[i] *= pre
            pre = pre*nums[i]
            sol[length-i-1] *= post
            post = post*nums[length-i-1]
        return(sol)