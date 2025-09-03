#date: 2025-09-03T16:56:36Z
#url: https://api.github.com/gists/b0b765e7b928ee5bca4589070e8e88d6
#owner: https://api.github.com/users/yuvalmoscovitz

def nextGreaterElement(nums: List[int]) -> List[int]:
    ans = [-1] * len(nums)
    stack = []
    for i in reversed(range(len(nums))):
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        if stack:
            ans[i] = stack[-1]
        stack.append(nums[i])
    return ans