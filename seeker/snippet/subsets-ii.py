#date: 2023-06-28T17:09:12Z
#url: https://api.github.com/gists/549b11771328fd28f9b75ff9f86226f1
#owner: https://api.github.com/users/markbrutx

def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()

    def backtrack(i, subset):
        if i == len(nums):
            res.append(subset[::])
            return

        # All subsets that include nums[i]
        subset.append(nums[i])
        backtrack(i + 1, subset)
        subset.pop()
        # All subsets that don't include nums[i]
        while i + 1 < len(nums) and nums[i] == nums[i + 1]:
            i += 1
        backtrack(i + 1, subset)

    backtrack(0, [])
    return res