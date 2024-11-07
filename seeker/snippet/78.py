#date: 2024-11-07T17:06:39Z
#url: https://api.github.com/gists/88cc9e240813fb6364276d8cc56e8196
#owner: https://api.github.com/users/nguyenhongson1902

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        used = [False] * len(nums)

        def backtrack(i, combination):
            # Goal
            if i == len(nums):
                res.append(combination.copy())
                return
            
            # Constraints
            if i < len(nums):
                if not used[i]:
                    used[i] = True
                    combination.append(nums[i])
                    backtrack(i + 1, combination) # first choice
                    combination.pop()
                    backtrack(i + 1, combination) # second choice
                    used[i] = False
                else:
                    i += 1
        
        backtrack(0, [])
        return res
        # Time: O(2^n), n = len(nums)
        # Space: O(n)
