#date: 2021-12-24T16:37:11Z
#url: https://api.github.com/gists/1081732574bf9654ae62a8381713dcbe
#owner: https://api.github.com/users/tseng1026

class Solution:
    # time - O(n) / space - O(n)
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        length = len(nums)
        nums = nums + nums
        
        monotonic = []
        next_greater = [-1] * length
        for index, num in enumerate(nums):
            if len(monotonic) != 0:
                while monotonic:
                    temp = nums[monotonic[-1]]
                    if num <= temp: break
                    if num > temp:
                        next_greater[monotonic[-1] % length] = num
                        monotonic.pop()
            monotonic.append(index)
        return next_greater